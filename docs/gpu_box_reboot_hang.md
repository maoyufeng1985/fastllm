# 远程主机 reboot 卡住：根因与 `tools/safe_reboot.sh`

四卡 V100 机器（Rocky 9、systemd 252、iTCO 硬件 watchdog）执行 reboot 时，关机流程停在
"等进程"，直到硬件 watchdog 超时才真正重启。本文记录根因，以及为此加的脚本。

## 1. 症状（观测）

console 上关机阶段的原文：

```
systemd-shutdown[1]: Using hardware watchdog 'iTCO_wdt', version 2, device /dev/watchdog0
systemd-shutdown[1]: Watchdog running with a timeout of 10min.
systemd-shutdown[1]: Sending SIGTERM to remaining processes...
systemd-shutdown[1]: Sending SIGKILL to remaining processes...
systemd-shutdown[1]: Waiting for process: 2452386 (python3), 2464417 (bash)
systemd-shutdown[1]: Sending SIGKILL to PID 2452386 (python3), 2464417 (bash)
systemd-shutdown[1]: Waiting for process: 2452386 (python3), 2464417 (bash)
```

读法：`SIGKILL` **已经发出**，但下一次轮询这两个进程**还在**。这不是"没杀"，是"杀不动"。

**这份截图对应哪一次事件（算出来的，已闭合）**：截图里的 uptime 是 `58161.9 s`。
本会话实测读到的 dmesg 里，`Xid 79 GPU has fallen off the bus` 的时间戳是 `57511.3 s`，
两者相差 `650.6 s`，即**掉卡之后约 10 分 50 秒**，console 仍停在这里。事后 `uptime -s`
显示机器在 **09:39:51** 起来。三个点连起来就是：掉卡约 09:25，关机流程约 09:29 开始等进程，
到 09:35:53 仍在等，硬件 watchdog 到点后 09:39:51 复位完成。
**所以"卡住"的真实时长就是那把 10 分钟。**

## 2. 根因链

1. **观测**：被等的两个进程是 `python3` 和一个 `bash`。
2. **推断（依据：本会话早前的 `ps` 记录，加连续 pid 关系）**：`2452386` 是
   `timeout 300 env FASTLLM_TP_AR_SIDE_STREAM=1 python3 -m ftllm.cli benchmark ...`
   的 python 子进程（pid 2452385 是那个 `timeout` 包装），`2464417` 是它的 bash 侧。
   逐 pid 的父子关系**当时没留进审计**（写审计的那段脚本有语法错误、没落盘），
   所以这一步是推断而非观测。但它不影响结论，脚本处理的是**这一类进程**，不是这两个号。
3. **观测**：`SIGKILL` 之后进程仍存活 → 它们处于 **D 状态（不可中断睡眠）**。
   `SIGKILL` 对 D 状态任务**无效**，它只能在被等待的内核操作返回后才会死。这台机器上
   这个内核操作就是 NVIDIA 驱动里的调用。同一现象本会话实测过两次，见审计：
   - `.audit/sm70-longctx-kv.tsv:97`：R2 那一臂跑满 400 s（正常 ~40 s），四卡 util 全 0%
     但各占 5.5 GB，python 37 线程中主线程等 10 个卡在 `futex_do_wait`、3 个自旋，
     属主机侧锁死，设备无新 Xid。
   - `.audit/sm70-longctx-kv.tsv:92`：另一场四卡 `Xid 79`（fallen off the bus）+
     `Xid 154`（需重置）+ `pcieport AER`，进程随即无法正常退出。
   两次都留下"进程杀不掉、显存不还"的现场，正是关机流程要等的状态。
4. **观测**：`systemctl show -p RebootWatchdogUSec` = **10min**（systemd 默认，
   `/etc/systemd/system.conf:35` 的 `#RebootWatchdogSec=10min` 是注释状态，即未覆盖）。
   **这就是那把"10 分钟"**：`systemd-shutdown` 反复等这些进程，同时硬件 watchdog 在计时。
5. **结论**：机器**不是死机**。它在等一个永远杀不掉的进程，10 分钟后由硬件 watchdog
   强制复位。所以症状是"卡住"，实际是"慢 10 分钟"。

**根因一句话**：**在 GPU 测试进程还活着（甚至已经卡在内核里）的时候发起了 reboot。**

## 3. 修在哪两层

- **主机侧（脚本的职责）**：rebooot 之前先把这类进程停干净并**验证**；停不掉的要明确报出来，
  而不是照样发 reboot 然后卡在关机里。
- **内核侧（兜底）**：把 `RebootWatchdogSec` 从默认 10min 降到 1-2min，把最坏情况的等待
  封顶。这一步只缩短上限，不消除卡住。

**为什么这个旋钮正好对症（本机 manpage，`systemd-system.conf`）**：`RebootWatchdogSec=`
是"重启时的硬件 watchdog 配置……作为安全网，保证即使干净重启超时、重启也一定会发生"，
并且它的超时**只作用于重启的第二阶段**，即"所有常规服务已终止、PID 1 已被
`systemd-shutdown` 取代之后"。**我们卡住的位置正是这个第二阶段**，所以它是对的旋钮。
（交叉验证：配置里 `#RebootWatchdogSec=10min` 是注释态、运行时
`RebootWatchdogUSec=10min`，与 console 上那句 `timeout of 10min` 三者一致。）

**落地细节（观测）**：本机 `/etc/systemd/system.conf.d/` **不存在**，写 drop-in 前要先
`mkdir -p`。

**现状（观测）**：`/proc/sys/kernel/sysrq` = `16`，只打开了 `sync` 那一位（`v & 16` 为真、
`v & 128` 为假）；`b`（重启）需要第 128 位，**当前被拒**。所以"直接用 sysrq 强制重启"这条
兜底在本机默认不可用，紧急路径必须先抬掩码再触发。两个节点
（`/proc/sys/kernel/sysrq`、`/proc/sysrq-trigger`）**当前均可写**，所以 root 下这条路可行。
触发机制已实测过一次（只用了当前允许的 `sync` 功能，没碰重启位）：`echo s >
/proc/sysrq-trigger` 后 dmesg 出现 `sysrq: Emergency Sync`，证明该节点确实接收写入、
被拒的只有掩码没打开的功能。脚本在触发重启前先 `sync`，把立即复位的数据风险压到最低。

## 4. 交付：`tools/safe_reboot.sh`

默认**只报告、不动手**；要动手必须显式给动作开关。

```
tools/safe_reboot.sh                # 只报告：目标进程、GPU 占用、D 状态进程、watchdog 值
tools/safe_reboot.sh --clean        # 执行清理（SIGTERM → SIGKILL），不重启
tools/safe_reboot.sh --reboot       # 清理 + 重启；若仍有杀不掉的进程，走紧急路径
```

选项：`--grace N`（TERM 后等多久，默认 20s）、`--kill-wait N`（KILL 后等多久，默认 10s）、
`--pattern P`（追加匹配，可重复）、`--watchdog-sec N`（写 drop-in 降 `RebootWatchdogSec`）、
`--log FILE`、`--hard`、`-h`。

目标集合 = **GPU 占用者**（`nvidia-smi --query-compute-apps`）∪ 固定模式
（`ftllm.cli`、`nsys profile`、`gpu_watchdog.sh`、`nccl_bench`、`fi_vs_nccl`）∪ 调用者
`--pattern`；**永远排除**自身、父进程、自身的全部祖先，以及命令行里含 `safe_reboot.sh`
的进程。选择模式时读 `/proc/<pid>/cmdline`，不用可能匹配到自己的 `pgrep -f`
（本仓库有过自匹配的教训）。

退出码：`0` 干净、`1` 仍有杀不掉的进程、`2` 用法错、`3` 非 root。

## 5. 验证方式

- 报告模式**不改任何状态**：前后 `ps` 快照一致、未发任何信号、`/etc` 无改动。
- 杀进程路径用**假目标**演练（`exec -a` 起的 `sleep`），不用生产模式串，
  本机随时可能有别的代理的 benchmark 在跑，不能拿它当靶子。
- 绝不为了测试真的 reboot，也不走紧急路径。

## 6. 一句话给下一个人

**这台机器上 reboot 之前，先跑 `tools/safe_reboot.sh`。** 它告诉你有没有杀不掉的 GPU 进程；
有的话，直接 reboot 会卡 10 分钟。想把这个上限也压掉，就顺手把 `RebootWatchdogSec` 设成
1-2 分钟。
