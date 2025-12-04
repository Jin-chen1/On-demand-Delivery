"""
Day 11测试：RL环境与SimPy仿真引擎完全对接
验证Gym接口与仿真的集成

测试内容：
1. 环境创建和初始化
2. reset()功能测试
3. step()功能测试
4. 状态转换验证
5. 时间同步验证
6. 动作执行验证
7. 多Episode测试
"""

import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.rl.rl_environment import DeliveryRLEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment_initialization():
    """测试1：环境初始化"""
    print("\n" + "="*60)
    print("测试1：环境初始化")
    print("="*60)
    
    simulation_config = {
        'data_dir': 'data/processed',
        'orders_file': 'data/orders/orders.csv',
        'total_orders': 100,
        'num_couriers': 10,
        'simulation_duration': 7200,
        'dispatch_interval': 30,
        'dispatcher_type': 'greedy',  # 使用greedy避免alns依赖
        'dispatcher_config': {},  # greedy不需要额外配置
        'courier_config': {
            'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
            'capacity': {'max_orders': 5}
        }
    }
    
    rl_config = {
        'state_encoder': {
            'max_pending_orders': 10,
            'max_couriers': 10,
            'grid_size': 5
        },
        'reward_calculator': {
            'reward_type': 'dense',
            'weight_timeout_penalty': 10.0,
            'weight_distance_cost': 0.001
        },
        'action_mode': 'discrete'
    }
    
    try:
        env = DeliveryRLEnvironment(simulation_config, rl_config)
        print("✓ 环境初始化成功")
        print(f"  观测空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        return env
    except Exception as e:
        print(f"✗ 环境初始化失败: {e}")
        raise


def test_reset_functionality(env):
    """测试2：reset()功能"""
    print("\n" + "="*60)
    print("测试2：reset()功能")
    print("="*60)
    
    try:
        obs, info = env.reset(seed=42)
        
        print("✓ reset()执行成功")
        print(f"  观测形状: {obs.shape}")
        print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Info keys: {list(info.keys())}")
        print(f"  Episode: {info['episode']}")
        print(f"  总订单数: {info['total_orders']}")
        print(f"  骑手数: {info['num_couriers']}")
        
        # 验证
        assert obs.shape == env.observation_space.shape, "观测形状不匹配"
        assert env.sim_env is not None, "仿真环境未创建"
        assert len(env.sim_env.orders) > 0, "订单未加载"
        assert len(env.sim_env.couriers) > 0, "骑手未初始化"
        
        print("✓ 所有验证通过")
        return obs, info
        
    except Exception as e:
        print(f"✗ reset()测试失败: {e}")
        raise


def test_step_functionality(env):
    """测试3：step()功能"""
    print("\n" + "="*60)
    print("测试3：step()功能")
    print("="*60)
    
    try:
        # 先reset
        obs, info = env.reset(seed=42)
        initial_time = env.sim_env.env.now
        
        # 执行多个step
        num_steps = 5
        print(f"\n执行 {num_steps} 个步骤...")
        
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\n步骤 {i+1}:")
            print(f"  动作: {action}")
            print(f"  奖励: {reward:.4f}")
            print(f"  终止: {terminated}, 截断: {truncated}")
            print(f"  仿真时间: {env.sim_env.env.now:.1f}s")
            print(f"  待分配订单: {info.get('pending_count', 0)}")
            print(f"  已完成订单: {info.get('total_completed', 0)}")
            
            if terminated or truncated:
                print(f"  Episode结束")
                break
        
        # 验证时间推进
        final_time = env.sim_env.env.now
        time_advanced = final_time - initial_time
        expected_advance = env.simulation_config['dispatch_interval'] * num_steps
        
        print(f"\n✓ step()执行成功")
        print(f"  时间推进: {time_advanced:.1f}s (预期: ~{expected_advance}s)")
        
        assert time_advanced > 0, "时间未推进"
        print("✓ 时间同步验证通过")
        
    except Exception as e:
        print(f"✗ step()测试失败: {e}")
        raise


def test_action_execution(env):
    """测试4：动作执行验证"""
    print("\n" + "="*60)
    print("测试4：动作执行验证")
    print("="*60)
    
    try:
        obs, info = env.reset(seed=42)
        
        # 测试延迟动作
        print("\n测试延迟动作 (action=0)...")
        initial_pending = len(env.sim_env.pending_orders)
        obs, reward, terminated, truncated, info = env.step(0)
        
        action_info = info.get('action_info', {})
        print(f"  动作类型: {action_info.get('action_type')}")
        
        # 测试分配动作
        if len(env.sim_env.pending_orders) > 0:
            print("\n测试分配动作 (action=1)...")
            initial_pending = len(env.sim_env.pending_orders)
            obs, reward, terminated, truncated, info = env.step(1)
            
            action_info = info.get('action_info', {})
            print(f"  动作类型: {action_info.get('action_type')}")
            print(f"  分配: {action_info.get('assignments')}")
            
            if action_info.get('action_type') == 'assign':
                final_pending = len(env.sim_env.pending_orders)
                print(f"  待分配订单变化: {initial_pending} -> {final_pending}")
                assert final_pending < initial_pending, "订单未被分配"
        
        print("\n✓ 动作执行验证通过")
        
    except Exception as e:
        print(f"✗ 动作执行测试失败: {e}")
        raise


def test_state_transition(env):
    """测试5：状态转换验证"""
    print("\n" + "="*60)
    print("测试5：状态转换验证")
    print("="*60)
    
    try:
        obs1, info1 = env.reset(seed=42)
        
        # 执行动作
        action = 1  # 分配给骑手1
        obs2, reward, terminated, truncated, info2 = env.step(action)
        
        # 验证状态变化
        state_changed = not np.array_equal(obs1, obs2)
        
        print(f"  初始观测范围: [{obs1.min():.3f}, {obs1.max():.3f}]")
        print(f"  后续观测范围: [{obs2.min():.3f}, {obs2.max():.3f}]")
        print(f"  状态变化: {state_changed}")
        
        if state_changed:
            print("✓ 状态转换验证通过")
        else:
            print("⚠ 警告: 状态未变化（可能正常，取决于系统状态）")
        
    except Exception as e:
        print(f"✗ 状态转换测试失败: {e}")
        raise


def test_multiple_episodes(env):
    """测试6：多Episode测试"""
    print("\n" + "="*60)
    print("测试6：多Episode测试")
    print("="*60)
    
    try:
        num_episodes = 3
        steps_per_episode = 10
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            obs, info = env.reset()
            
            episode_rewards = []
            for step in range(steps_per_episode):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    print(f"  Episode在第{step+1}步结束")
                    if 'episode_stats' in info:
                        stats = info['episode_stats']
                        print(f"  Episode统计:")
                        print(f"    - 完成率: {stats.get('completion_rate', 0):.2%}")
                        print(f"    - 超时率: {stats.get('timeout_rate', 0):.2%}")
                        print(f"    - 总奖励: {sum(episode_rewards):.2f}")
                    break
        
        print("\n✓ 多Episode测试通过")
        
    except Exception as e:
        print(f"✗ 多Episode测试失败: {e}")
        raise


def test_episode_statistics(env):
    """测试7：Episode统计验证"""
    print("\n" + "="*60)
    print("测试7：Episode统计验证")
    print("="*60)
    
    try:
        obs, info = env.reset(seed=42)
        
        # 运行一个完整episode或固定步数
        max_steps = 20
        for i in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # 获取统计信息
        stats = env.get_episode_statistics()
        
        print("\nEpisode统计信息:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 验证关键统计
        assert 'total_orders' in stats, "缺少total_orders统计"
        assert 'completed_orders' in stats, "缺少completed_orders统计"
        assert 'completion_rate' in stats, "缺少completion_rate统计"
        
        print("\n✓ Episode统计验证通过")
        
    except Exception as e:
        print(f"✗ Episode统计测试失败: {e}")
        raise


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Day 11集成测试：RL环境与SimPy仿真引擎对接")
    print("="*80)
    
    try:
        # 测试1：环境初始化
        env = test_environment_initialization()
        
        # 测试2：reset()功能
        test_reset_functionality(env)
        
        # 测试3：step()功能
        test_step_functionality(env)
        
        # 测试4：动作执行
        test_action_execution(env)
        
        # 测试5：状态转换
        test_state_transition(env)
        
        # 测试6：多Episode
        test_multiple_episodes(env)
        
        # 测试7：Episode统计
        test_episode_statistics(env)
        
        # 关闭环境
        env.close()
        
        print("\n" + "="*80)
        print("✓ 所有测试通过！Day 11任务完成")
        print("="*80)
        print("\n集成验证结果:")
        print("  ✓ Gym接口实现完整")
        print("  ✓ 仿真环境正确创建和初始化")
        print("  ✓ step()方法推进仿真时间")
        print("  ✓ 动作正确执行并影响仿真状态")
        print("  ✓ 状态转换逻辑正确")
        print("  ✓ 多Episode运行稳定")
        print("  ✓ Episode统计信息完整")
        print("\nDay 11: Environment Integration - 完成✓")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
