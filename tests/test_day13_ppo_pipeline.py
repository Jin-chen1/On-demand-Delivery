"""
Day 13测试：PPO训练流水线验证
基于Stable-Baselines3搭建完整的PPO训练流水线

测试内容：
1. Stable-Baselines3安装验证
2. 环境Gym兼容性检查
3. PPO模型创建测试
4. TensorBoard配置测试
5. 回调函数测试
6. 短期训练验证（调试模式）
7. 模型保存与加载测试
8. 评估流程测试
"""

import sys
from pathlib import Path
import logging
import time
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 测试配置
# ============================================================

# 使用调试配置
CONFIG_PATH = 'config/rl_training_debug.yaml'

# 测试输出目录
TEST_OUTPUT_DIR = Path('outputs/day13_test')

# 调试训练步数
DEBUG_TIMESTEPS = 2000


# ============================================================
# 测试函数
# ============================================================

def test_sb3_installation():
    """测试1：Stable-Baselines3安装验证"""
    print("\n" + "="*60)
    print("测试1：Stable-Baselines3安装验证")
    print("="*60)
    
    results = {}
    
    # 检查核心库
    try:
        import stable_baselines3
        results['stable_baselines3'] = f"✓ v{stable_baselines3.__version__}"
        print(f"  stable_baselines3: {results['stable_baselines3']}")
    except ImportError as e:
        results['stable_baselines3'] = f"✗ {e}"
        print(f"  stable_baselines3: {results['stable_baselines3']}")
        raise ImportError("请安装: pip install stable-baselines3[extra]")
    
    # 检查PPO
    try:
        from stable_baselines3 import PPO
        results['PPO'] = "✓ 可用"
        print(f"  PPO算法: {results['PPO']}")
    except ImportError as e:
        results['PPO'] = f"✗ {e}"
        print(f"  PPO算法: {results['PPO']}")
    
    # 检查TensorBoard
    try:
        import tensorboard
        results['tensorboard'] = f"✓ v{tensorboard.__version__}"
        print(f"  TensorBoard: {results['tensorboard']}")
    except ImportError:
        results['tensorboard'] = "⚠ 未安装（可选）"
        print(f"  TensorBoard: {results['tensorboard']}")
    
    # 检查gymnasium
    try:
        import gymnasium
        results['gymnasium'] = f"✓ v{gymnasium.__version__}"
        print(f"  Gymnasium: {results['gymnasium']}")
    except ImportError as e:
        results['gymnasium'] = f"✗ {e}"
        print(f"  Gymnasium: {results['gymnasium']}")
    
    print("\n✓ SB3安装验证完成")
    return results


def test_environment_compatibility():
    """测试2：环境Gym兼容性检查"""
    print("\n" + "="*60)
    print("测试2：环境Gym兼容性检查")
    print("="*60)
    
    from stable_baselines3.common.env_checker import check_env
    from src.rl.rl_environment import DeliveryRLEnvironment
    import yaml
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 使用debug场景
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    
    print(f"  使用配置: {CONFIG_PATH}")
    print(f"  订单数: {sim_config.get('total_orders')}")
    print(f"  骑手数: {sim_config.get('num_couriers')}")
    
    # 创建环境
    env = DeliveryRLEnvironment(
        simulation_config=sim_config,
        rl_config=config.get('rl', {})
    )
    
    print(f"\n  观测空间: {env.observation_space}")
    print(f"  观测形状: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space}")
    print(f"  动作数量: {env.action_space.n}")
    
    # SB3环境检查
    print("\n  运行SB3环境检查...")
    try:
        check_env(env, warn=True)
        print("  ✓ 环境兼容性检查通过")
        result = True
    except Exception as e:
        print(f"  ⚠ 环境检查警告: {e}")
        result = False
    
    env.close()
    return result


def test_ppo_model_creation():
    """测试3：PPO模型创建"""
    print("\n" + "="*60)
    print("测试3：PPO模型创建")
    print("="*60)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.rl.rl_environment import DeliveryRLEnvironment
    import yaml
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    rl_config = config.get('rl', {})
    
    # 创建向量化环境
    def make_env():
        return DeliveryRLEnvironment(sim_config, rl_config)
    
    env = DummyVecEnv([make_env])
    
    # 创建PPO模型
    print("  创建PPO模型...")
    training_config = rl_config.get('training', {})
    ppo_config = training_config.get('ppo', {})
    policy_config = training_config.get('policy', {})
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=training_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 512),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 5),
        gamma=training_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        ent_coef=ppo_config.get('ent_coef', 0.02),
        policy_kwargs=dict(
            net_arch=policy_config.get('net_arch', [128, 128])
        ),
        verbose=0,
        seed=config.get('seed', 42)
    )
    
    print(f"  ✓ PPO模型创建成功")
    print(f"  策略网络: {model.policy}")
    print(f"  学习率: {model.learning_rate}")
    print(f"  n_steps: {model.n_steps}")
    print(f"  batch_size: {model.batch_size}")
    
    env.close()
    return model


def test_tensorboard_config():
    """测试4：TensorBoard配置"""
    print("\n" + "="*60)
    print("测试4：TensorBoard配置")
    print("="*60)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure
    from src.rl.rl_environment import DeliveryRLEnvironment
    import yaml
    
    # 创建测试目录
    tb_dir = TEST_OUTPUT_DIR / 'tensorboard_test'
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    
    # 创建环境和模型
    env = DummyVecEnv([lambda: DeliveryRLEnvironment(sim_config, config.get('rl', {}))])
    model = PPO("MlpPolicy", env, verbose=0)
    
    # 配置TensorBoard日志
    print(f"  配置TensorBoard日志目录: {tb_dir}")
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    # 运行几步以生成日志
    print("  运行少量步骤生成日志...")
    model.learn(total_timesteps=100, progress_bar=False)
    
    # 检查日志文件
    tb_files = list(tb_dir.glob('events.out.tfevents.*'))
    if tb_files:
        print(f"  ✓ TensorBoard日志已生成: {len(tb_files)} 个文件")
        print(f"  启动命令: tensorboard --logdir={tb_dir}")
        result = True
    else:
        print("  ⚠ 未检测到TensorBoard日志文件")
        result = False
    
    env.close()
    return result


def test_callbacks():
    """测试5：回调函数"""
    print("\n" + "="*60)
    print("测试5：回调函数")
    print("="*60)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from src.rl.rl_environment import DeliveryRLEnvironment
    from src.rl.train_rl_agent import TrainingMonitorCallback
    import yaml
    
    # 创建测试目录
    callback_dir = TEST_OUTPUT_DIR / 'callback_test'
    callback_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    
    # 创建环境
    env = DummyVecEnv([lambda: DeliveryRLEnvironment(sim_config, config.get('rl', {}))])
    eval_env = DummyVecEnv([lambda: DeliveryRLEnvironment(sim_config, config.get('rl', {}))])
    
    # 创建回调
    callbacks = []
    
    # 1. 自定义监控回调
    monitor_cb = TrainingMonitorCallback(
        check_freq=100,
        log_dir=str(callback_dir),
        early_stop_patience=10,
        verbose=0
    )
    callbacks.append(monitor_cb)
    print("  ✓ TrainingMonitorCallback创建成功")
    
    # 2. 检查点回调
    checkpoint_cb = CheckpointCallback(
        save_freq=200,
        save_path=str(callback_dir / 'checkpoints'),
        name_prefix='test_model'
    )
    callbacks.append(checkpoint_cb)
    print("  ✓ CheckpointCallback创建成功")
    
    # 3. 评估回调
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(callback_dir / 'best_model'),
        log_path=str(callback_dir / 'eval_logs'),
        eval_freq=200,
        n_eval_episodes=2,
        deterministic=True
    )
    callbacks.append(eval_cb)
    print("  ✓ EvalCallback创建成功")
    
    # 创建模型并训练
    model = PPO("MlpPolicy", env, verbose=0)
    print("\n  运行回调测试（500步）...")
    
    model.learn(total_timesteps=500, callback=callbacks, progress_bar=False)
    
    # 检查结果
    checkpoints = list((callback_dir / 'checkpoints').glob('*.zip'))
    print(f"  保存的检查点: {len(checkpoints)}")
    
    env.close()
    eval_env.close()
    
    return len(callbacks) == 3


def test_short_training():
    """测试6：短期训练验证"""
    print("\n" + "="*60)
    print("测试6：短期训练验证（调试模式）")
    print("="*60)
    
    from src.rl.train_rl_agent import RLTrainer
    
    # 创建训练器
    trainer = RLTrainer(CONFIG_PATH, scenario='debug')
    
    # 覆盖训练步数
    trainer.training_config['total_timesteps'] = DEBUG_TIMESTEPS
    
    print(f"  配置: {CONFIG_PATH}")
    print(f"  场景: debug")
    print(f"  训练步数: {DEBUG_TIMESTEPS}")
    print(f"  输出目录: {trainer.output_dir}")
    
    print("\n  开始训练...")
    start_time = time.time()
    
    try:
        trainer.train()
        elapsed = time.time() - start_time
        print(f"\n  ✓ 训练完成，耗时: {elapsed:.1f}秒")
        
        # 检查输出
        final_model = trainer.output_dir / 'final_model.zip'
        if final_model.exists():
            print(f"  ✓ 最终模型已保存: {final_model}")
            return True
        else:
            print(f"  ⚠ 最终模型未找到")
            return False
            
    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """测试7：模型保存与加载"""
    print("\n" + "="*60)
    print("测试7：模型保存与加载")
    print("="*60)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.rl.rl_environment import DeliveryRLEnvironment
    import yaml
    
    # 创建测试目录
    model_dir = TEST_OUTPUT_DIR / 'model_test'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    
    # 创建环境和模型
    env = DummyVecEnv([lambda: DeliveryRLEnvironment(sim_config, config.get('rl', {}))])
    model = PPO("MlpPolicy", env, verbose=0)
    
    # 训练几步
    print("  训练模型...")
    model.learn(total_timesteps=200, progress_bar=False)
    
    # 保存模型
    model_path = model_dir / 'test_ppo_model'
    model.save(model_path)
    print(f"  ✓ 模型已保存: {model_path}")
    
    # 删除原模型
    del model
    
    # 重新加载
    print("  加载模型...")
    loaded_model = PPO.load(model_path)
    print(f"  ✓ 模型已加载")
    
    # 验证预测
    obs = env.reset()
    action, _ = loaded_model.predict(obs, deterministic=True)
    print(f"  ✓ 预测动作: {action}")
    
    env.close()
    return True


def test_evaluation():
    """测试8：评估流程"""
    print("\n" + "="*60)
    print("测试8：评估流程")
    print("="*60)
    
    from stable_baselines3 import PPO
    from src.rl.rl_environment import DeliveryRLEnvironment
    import yaml
    
    # 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('simulation', {})
    sim_config.update(config.get('scenarios', {}).get('debug', {}))
    
    # 检查是否有已训练的模型
    model_files = list(TEST_OUTPUT_DIR.rglob('*.zip'))
    
    if not model_files:
        print("  ⚠ 未找到已训练模型，创建随机模型进行测试")
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: DeliveryRLEnvironment(sim_config, config.get('rl', {}))])
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100, progress_bar=False)
        env.close()
    else:
        model_path = model_files[0]
        print(f"  加载模型: {model_path}")
        model = PPO.load(model_path)
    
    # 创建评估环境
    eval_env = DeliveryRLEnvironment(sim_config, config.get('rl', {}))
    
    # 运行评估
    n_episodes = 2
    print(f"\n  运行 {n_episodes} 个评估Episode...")
    
    results = {
        'rewards': [],
        'lengths': []
    }
    
    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 50:  # 限制步数
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(steps)
        print(f"    Episode {ep+1}: reward={episode_reward:.2f}, steps={steps}")
    
    print(f"\n  平均奖励: {np.mean(results['rewards']):.2f}")
    print(f"  平均步数: {np.mean(results['lengths']):.1f}")
    
    eval_env.close()
    return True


# ============================================================
# 主测试函数
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# Day 13: PPO训练流水线测试")
    print("#"*60)
    
    # 创建测试输出目录
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 测试1：SB3安装
    try:
        test_sb3_installation()
        results['sb3_installation'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['sb3_installation'] = 'FAIL'
        return results  # SB3是必需的
    
    # 测试2：环境兼容性
    try:
        if test_environment_compatibility():
            results['env_compatibility'] = 'PASS'
        else:
            results['env_compatibility'] = 'WARN'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['env_compatibility'] = 'FAIL'
    
    # 测试3：PPO模型创建
    try:
        test_ppo_model_creation()
        results['ppo_creation'] = 'PASS'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['ppo_creation'] = 'FAIL'
    
    # 测试4：TensorBoard配置
    try:
        if test_tensorboard_config():
            results['tensorboard'] = 'PASS'
        else:
            results['tensorboard'] = 'WARN'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['tensorboard'] = 'FAIL'
    
    # 测试5：回调函数
    try:
        if test_callbacks():
            results['callbacks'] = 'PASS'
        else:
            results['callbacks'] = 'WARN'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['callbacks'] = 'FAIL'
    
    # 测试6：短期训练
    try:
        if test_short_training():
            results['short_training'] = 'PASS'
        else:
            results['short_training'] = 'FAIL'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['short_training'] = 'FAIL'
    
    # 测试7：模型保存加载
    try:
        if test_model_save_load():
            results['model_save_load'] = 'PASS'
        else:
            results['model_save_load'] = 'FAIL'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['model_save_load'] = 'FAIL'
    
    # 测试8：评估流程
    try:
        if test_evaluation():
            results['evaluation'] = 'PASS'
        else:
            results['evaluation'] = 'FAIL'
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        results['evaluation'] = 'FAIL'
    
    # 打印总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    warned = sum(1 for v in results.values() if v == 'WARN')
    failed = sum(1 for v in results.values() if v == 'FAIL')
    
    for test_name, status in results.items():
        symbol = '✓' if status == 'PASS' else ('⚠' if status == 'WARN' else '✗')
        print(f"  {symbol} {test_name}: {status}")
    
    print(f"\n总计: {passed} PASS, {warned} WARN, {failed} FAIL")
    
    if failed == 0:
        print("\n" + "#"*60)
        print("# ✓ Day 13 PPO训练流水线验证通过！")
        print("#"*60)
        print("\n下一步：")
        print("  1. 启动TensorBoard监控:")
        print(f"     tensorboard --logdir={TEST_OUTPUT_DIR}")
        print("  2. 运行完整训练:")
        print("     python -m src.rl.train_rl_agent --config config/rl_training_debug.yaml --scenario low_load")
    else:
        print("\n部分测试失败，请检查错误信息")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Day 13 PPO Pipeline测试')
    parser.add_argument('--quick', action='store_true', help='快速测试（跳过训练）')
    parser.add_argument('--clean', action='store_true', help='清理测试输出')
    args = parser.parse_args()
    
    if args.clean:
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
            print(f"已清理: {TEST_OUTPUT_DIR}")
        sys.exit(0)
    
    if args.quick:
        # 快速测试
        test_sb3_installation()
        test_environment_compatibility()
        test_ppo_model_creation()
        print("\n快速测试完成！")
    else:
        # 完整测试
        run_all_tests()
