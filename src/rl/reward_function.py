"""
奖励函数模块
定义RL Agent的奖励信号，引导学习最优派单策略

奖励设计原则（基于研究大纲）：
1. 稀疏奖励 vs 密集奖励权衡
2. 多目标平衡：超时惩罚、距离成本、等待时间
3. 长期价值：鼓励"预留运力"等前瞻性行为
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    奖励计算器
    根据决策后的状态变化计算奖励值
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化奖励计算器
        
        Args:
            config: 奖励函数配置
        """
        self.config = config or {}
        
        # 权重参数（可调节以平衡不同目标）
        # v4优化：方案B - 调整权重比例，增强关键信号
        self.weight_timeout_penalty = self.config.get('weight_timeout_penalty', 0.2)  # 增加2倍
        self.weight_distance_cost = self.config.get('weight_distance_cost', 0.001)    # 增加10倍
        self.weight_wait_time = self.config.get('weight_wait_time', 0.001)
        self.weight_completion_bonus = self.config.get('weight_completion_bonus', 0.5)
        self.weight_balanced_load = self.config.get('weight_balanced_load', 0.5)
        # 成功分配订单的即时奖励
        self.weight_assignment_bonus = self.config.get('weight_assignment_bonus', 0.2)  # 增加2倍
        
        # v4新增：方案A - 延迟派单惩罚权重
        self.weight_delay_penalty = self.config.get('weight_delay_penalty', 0.05)  # 每个待分配订单的惩罚
        self.weight_urgent_delay_penalty = self.config.get('weight_urgent_delay_penalty', 0.2)  # 紧急订单延迟惩罚
        
        # v4新增：方案C - 紧急订单优先奖励
        self.weight_urgent_assignment = self.config.get('weight_urgent_assignment', 0.5)  # 紧急订单分配奖励
        
        # v2新增：里程碑奖励（每完成一定数量订单给予额外奖励）
        self.weight_milestone_bonus = self.config.get('weight_milestone_bonus', 1.0)
        self.milestone_interval = self.config.get('milestone_interval', 10)
        self.completed_count_for_milestone = 0  # 用于跟踪里程碑
        
        # Day 23: 商家等餐时间惩罚权重
        self.weight_merchant_wait = self.config.get('weight_merchant_wait', 0.02)
        self.weight_busy_merchant_penalty = self.config.get('weight_busy_merchant_penalty', 2.0)
        
        # 奖励形式
        self.reward_type = self.config.get('reward_type', 'dense')  # 'dense' or 'sparse'
        
        # 统计
        self.total_rewards = 0.0
        self.step_count = 0
        
        # Day 23: 等餐时间统计
        self.total_merchant_wait_time = 0.0
        self.merchant_wait_count = 0
        
        logger.info("RewardCalculator初始化完成 (v4优化版)")
        logger.info(f"  奖励类型: {self.reward_type}")
        logger.info(f"  超时惩罚权重: {self.weight_timeout_penalty}")
        logger.info(f"  距离成本权重: {self.weight_distance_cost}")
        logger.info(f"  分配奖励权重: {self.weight_assignment_bonus}")
        logger.info(f"  延迟惩罚权重: {self.weight_delay_penalty} (紧急: {self.weight_urgent_delay_penalty})")
        logger.info(f"  紧急分配奖励: {self.weight_urgent_assignment}")
        logger.info(f"  里程碑奖励: {self.weight_milestone_bonus} (每{self.milestone_interval}单)")
    
    def calculate_step_reward(self,
                             state_before: Dict[str, Any],
                             action: Dict[str, Any],
                             state_after: Dict[str, Any],
                             info: Optional[Dict[str, Any]] = None) -> float:
        """
        计算单步奖励
        
        Args:
            state_before: 执行动作前的状态
            action: 执行的动作
            state_after: 执行动作后的状态
            info: 额外信息（如订单完成、超时等事件）
        
        Returns:
            奖励值
        """
        if self.reward_type == 'sparse':
            # 稀疏奖励：仅在订单完成时给予
            reward = self._calculate_sparse_reward(info)
        else:
            # 密集奖励：每步都给予反馈
            reward = self._calculate_dense_reward(state_before, action, state_after, info)
        
        # 更新统计
        self.total_rewards += reward
        self.step_count += 1
        
        return reward
    
    def _calculate_sparse_reward(self, info: Dict[str, Any]) -> float:
        """
        计算稀疏奖励（仅在关键事件时给予反馈）
        
        Args:
            info: 事件信息
        
        Returns:
            奖励值
        """
        reward = 0.0
        
        if not info:
            return reward
        
        # 订单完成事件
        if 'completed_orders' in info:
            for order_info in info['completed_orders']:
                if order_info.get('is_timeout', False):
                    # 超时完成：大惩罚
                    reward -= self.weight_timeout_penalty * 100
                else:
                    # 准时完成：正奖励
                    reward += self.weight_completion_bonus * 10
                    
                    # 额外奖励：提前完成（鼓励高效）
                    early_completion_time = order_info.get('slack_time', 0)
                    if early_completion_time > 0:
                        reward += min(early_completion_time / 60.0, 5.0)  # 最多5分奖励
        
        return reward
    
    def _calculate_dense_reward(self,
                                state_before: Dict[str, Any],
                                action: Dict[str, Any],
                                state_after: Dict[str, Any],
                                info: Optional[Dict[str, Any]] = None) -> float:
        """
        计算密集奖励（每步都给予反馈）
        Day 23: 扩展商家等餐时间惩罚
        
        奖励组成：
        1. 即时成本：分配导致的距离增加
        2. 时间窗惩罚：订单接近超时的风险增加
        3. 负载平衡：鼓励均衡分配
        4. 完成奖励：订单成功送达
        5. Day 23新增：商家等餐时间惩罚
        6. Day 23新增：避开繁忙商家奖励
        
        Returns:
            奖励值
        """
        reward = 0.0
        info = info or {}
        
        # 1. 距离成本（立即反馈）
        distance_increase = action.get('distance_increase', 0)
        reward -= self.weight_distance_cost * distance_increase
        
        # 2. 时间窗风险变化
        timeout_risk_before = self._calculate_system_timeout_risk(state_before)
        timeout_risk_after = self._calculate_system_timeout_risk(state_after)
        risk_change = timeout_risk_after - timeout_risk_before
        reward -= self.weight_timeout_penalty * risk_change
        
        # 3. 负载平衡奖励
        load_balance_before = self._calculate_load_balance(state_before)
        load_balance_after = self._calculate_load_balance(state_after)
        balance_improvement = load_balance_after - load_balance_before
        reward += self.weight_balanced_load * balance_improvement
        
        # 4. 等待时间惩罚（订单在队列中的时长）
        if 'pending_wait_time' in state_after:
            avg_wait_time = state_after['pending_wait_time']
            reward -= self.weight_wait_time * avg_wait_time
        
        # 5. 订单完成奖励（叠加稀疏奖励）
        if 'completed_orders' in info:
            reward += self._calculate_sparse_reward(info) / 10.0  # 缩小幅度
        
        # 6. 延迟派单惩罚（方案A）
        # 防止Agent滥用延迟派单来避免负奖励
        if action.get('action_type') == 'delay':
            pending_orders = state_after.get('pending_orders', [])
            pending_count = len(pending_orders)
            
            # 基础延迟惩罚：待分配订单越多，惩罚越大
            delay_penalty = self.weight_delay_penalty * pending_count
            reward -= delay_penalty
            
            # 额外惩罚：如果有紧急订单（30分钟内到期）却选择延迟
            current_time = state_after.get('current_time', 0)
            urgent_count = sum(1 for order in pending_orders 
                              if hasattr(order, 'latest_delivery_time') and 
                              order.latest_delivery_time - current_time < 1800)
            if urgent_count > 0:
                reward -= self.weight_urgent_delay_penalty * urgent_count
        
        # 7. 成功分配订单的即时奖励（鼓励及时分配而非延迟）
        if action.get('action_type') in ['assign', 'assign_fallback']:
            reward += self.weight_assignment_bonus
            # 如果使用回退机制分配，给予惩罚
            if action.get('action_type') == 'assign_fallback':
                reward -= self.weight_assignment_bonus * 0.3
            
            # Day 27修复: delay_justified补偿奖励应在分配时给予
            # 当订单之前被延迟过，且当前分配的骑手比延迟时的最佳骑手更优时，
            # 说明延迟决策是正确的，给予补偿奖励
            if action.get('delay_justified', False):
                reward += 2.0
                logger.debug("延迟派单被证明合理（分配时找到更优骑手），给予+2.0补偿奖励")
            
            # 方案C：紧急订单优先奖励
            # 如果分配的是紧急订单（30分钟内到期），给予额外奖励
            order_info = action.get('order_info', {})
            time_to_deadline = order_info.get('time_to_deadline', float('inf'))
            if time_to_deadline < 1800:  # 30分钟内
                urgency_bonus = self.weight_urgent_assignment * (1 - time_to_deadline / 1800)
                reward += urgency_bonus
        
        # 7.1 multi_discrete模式下的批量分配奖励
        if action.get('action_type') == 'multi_assign':
            assignments = action.get('assignments', [])
            current_time = state_after.get('current_time', 0)
            
            for assignment in assignments:
                # 每个成功分配都给予奖励
                reward += self.weight_assignment_bonus
                
                # 回退分配惩罚
                if assignment.get('is_fallback', False):
                    reward -= self.weight_assignment_bonus * 0.3
                
                # Day 27: delay_justified补偿奖励（multi_discrete模式）
                if assignment.get('delay_justified', False):
                    reward += 2.0
                    logger.debug(f"订单{assignment.get('order_id')}延迟合理，给予+2.0补偿奖励")
                
                # 方案C：紧急订单优先奖励
                order = assignment.get('order')
                if order and hasattr(order, 'latest_delivery_time'):
                    time_to_deadline = order.latest_delivery_time - current_time
                    if time_to_deadline < 1800:  # 30分钟内
                        urgency_bonus = self.weight_urgent_assignment * (1 - time_to_deadline / 1800)
                        reward += urgency_bonus
        
        # === Day 23: 商家等餐时间惩罚 ===
        reward += self._calculate_merchant_wait_penalty(state_after, action, info)
        
        # === v2新增：里程碑奖励 ===
        # 每完成一定数量订单给予额外奖励，强化学习信号
        if 'completed_orders' in info and info['completed_orders']:
            completed_count = len(info['completed_orders'])
            self.completed_count_for_milestone += completed_count
            
            # 检查是否达到新的里程碑
            milestones_reached = self.completed_count_for_milestone // self.milestone_interval
            previous_milestones = (self.completed_count_for_milestone - completed_count) // self.milestone_interval
            
            new_milestones = milestones_reached - previous_milestones
            if new_milestones > 0:
                milestone_reward = new_milestones * self.weight_milestone_bonus
                reward += milestone_reward
                logger.debug(f"里程碑奖励: +{milestone_reward:.1f} (已完成{self.completed_count_for_milestone}单)")
        
        return reward
    
    def _calculate_system_timeout_risk(self, state: Dict[str, Any]) -> float:
        """
        计算系统总体超时风险
        
        Args:
            state: 环境状态
        
        Returns:
            总体超时风险值
        """
        pending_orders = state.get('pending_orders', [])
        current_time = state.get('current_time', 0)
        
        if not pending_orders:
            return 0.0
        
        total_risk = 0.0
        for order in pending_orders:
            time_to_deadline = max(order.latest_delivery_time - current_time, 0)
            # 风险随剩余时间指数增长
            risk = np.exp(-time_to_deadline / 1800.0)  # 30分钟衰减
            total_risk += risk
        
        return total_risk / len(pending_orders)
    
    def _calculate_load_balance(self, state: Dict[str, Any]) -> float:
        """
        计算骑手负载平衡度
        
        负载越均衡，值越高（0-1）
        
        Args:
            state: 环境状态
        
        Returns:
            平衡度值
        """
        couriers = state.get('couriers', {})
        
        if not couriers:
            return 0.0
        
        loads = [len(c.current_route) for c in couriers.values()]
        
        if max(loads) == 0:
            return 1.0  # 所有骑手都空闲，完美平衡
        
        # 使用标准差衡量不平衡度
        std_dev = np.std(loads)
        mean_load = np.mean(loads)
        
        # 归一化
        if mean_load > 0:
            cv = std_dev / mean_load  # 变异系数
            balance_score = 1.0 / (1.0 + cv)  # 转换为[0, 1]分数
        else:
            balance_score = 1.0
        
        return balance_score
    
    def _calculate_merchant_wait_penalty(self,
                                         state: Dict[str, Any],
                                         action: Dict[str, Any],
                                         info: Dict[str, Any]) -> float:
        """
        Day 23: 计算商家等餐时间惩罚
        
        惩罚组成：
        1. 骑手实际等餐时间惩罚
        2. 分配给繁忙商家订单的惩罚
        3. 预估等餐时间过长的惩罚
        
        Args:
            state: 当前状态
            action: 执行的动作
            info: 额外信息
        
        Returns:
            商家等餐相关惩罚值（负数表示惩罚）
        """
        penalty = 0.0
        
        # 1. 骑手实际等餐时间惩罚
        # 注意：merchant_wait_events需要SimulationEnvironment.step()返回，目前是预留接口
        # 实现思路：当骑手到达商家后等待取餐时，记录等待时间作为事件
        if 'merchant_wait_events' in info:
            for wait_event in info['merchant_wait_events']:
                wait_time = wait_event.get('wait_time', 0)
                # 记录统计
                self.total_merchant_wait_time += wait_time
                self.merchant_wait_count += 1
                # 归一化惩罚（超过5分钟给予更大惩罚）
                if wait_time > 300:  # 5分钟
                    penalty -= self.weight_merchant_wait * wait_time * 1.5
                else:
                    penalty -= self.weight_merchant_wait * wait_time
        
        # 2. 分配给繁忙商家订单的惩罚
        # 支持discrete模式的单个分配和multi_discrete模式的批量分配
        merchants = state.get('merchants', {})
        
        # 收集所有分配的订单
        orders_to_check = []
        if action.get('action_type') == 'assign':
            assigned_order = action.get('order', None)
            if assigned_order:
                orders_to_check.append(assigned_order)
        elif action.get('action_type') == 'multi_assign':
            # multi_discrete模式：从assignments中提取订单
            for assignment in action.get('assignments', []):
                assigned_order = assignment.get('order', None)
                if assigned_order:
                    orders_to_check.append(assigned_order)
        
        # 检查每个分配的订单是否分配给繁忙商家
        for assigned_order in orders_to_check:
            if merchants:
                merchant_id = getattr(assigned_order, 'merchant_id', None)
                if merchant_id and merchant_id in merchants:
                    merchant = merchants[merchant_id]
                    utilization = merchant.get_utilization() if hasattr(merchant, 'get_utilization') else 0.0
                    
                    # 高利用率商家惩罚（超过80%利用率）
                    if utilization > 0.8:
                        penalty -= self.weight_busy_merchant_penalty * (utilization - 0.8) * 10
                    
                    # 超负荷商家惩罚
                    if hasattr(merchant, 'status'):
                        from ..simulation.entities import MerchantStatus
                        if merchant.status == MerchantStatus.OVERLOADED:
                            penalty -= self.weight_busy_merchant_penalty * 5.0
        
        # 3. 预估等餐时间过长的惩罚
        pending_orders = state.get('pending_orders', [])
        merchants = state.get('merchants', {})
        
        if pending_orders and merchants:
            high_wait_count = 0
            for order in pending_orders:
                merchant_id = getattr(order, 'merchant_id', None)
                if merchant_id and merchant_id in merchants:
                    merchant = merchants[merchant_id]
                    est_wait = merchant.estimate_wait_time() if hasattr(merchant, 'estimate_wait_time') else 0
                    # 预估等餐超过10分钟的订单
                    if est_wait > 600:
                        high_wait_count += 1
            
            # 系统中高等餐时间订单过多的惩罚
            if high_wait_count > 5:
                penalty -= self.weight_merchant_wait * (high_wait_count - 5) * 2.0
        
        return penalty
    
    def calculate_episode_reward(self, episode_info: Dict[str, Any]) -> float:
        """
        计算整个Episode的总奖励（用于评估）
        
        注意：目前此方法未在主流程中使用。
        真正的Episode统计由DeliveryRLEnvironment.get_episode_statistics()负责。
        此方法可用于post-hoc评估或自定义评估场景。
        
        TODO: 考虑在evaluate_model()中使用此方法，或删除以避免"两套Episode指标"的混淆。
        
        Args:
            episode_info: Episode结束时的统计信息
        
        Returns:
            总奖励值
        """
        total_reward = 0.0
        
        # 基于最终指标计算
        total_orders = episode_info.get('total_orders', 1)
        completed_orders = episode_info.get('completed_orders', 0)
        timeout_orders = episode_info.get('timeout_orders', 0)
        total_distance = episode_info.get('total_distance', 0)
        avg_service_time = episode_info.get('avg_service_time', 0)
        
        # 完成率奖励
        completion_rate = completed_orders / max(total_orders, 1)
        total_reward += completion_rate * 1000
        
        # 超时率惩罚
        timeout_rate = timeout_orders / max(total_orders, 1)
        total_reward -= timeout_rate * self.weight_timeout_penalty * 100
        
        # 总距离惩罚
        total_reward -= total_distance * self.weight_distance_cost
        
        # 平均服务时间惩罚
        total_reward -= avg_service_time * self.weight_wait_time / 10.0
        
        return total_reward
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """
        获取奖励统计信息
        Day 23: 扩展商家等餐时间统计
        
        Returns:
            统计字典
        """
        avg_merchant_wait = (self.total_merchant_wait_time / max(self.merchant_wait_count, 1))
        
        return {
            'total_rewards': self.total_rewards,
            'step_count': self.step_count,
            'average_reward': self.total_rewards / max(self.step_count, 1),
            # Day 23: 商家等餐统计
            'total_merchant_wait_time': self.total_merchant_wait_time,
            'merchant_wait_count': self.merchant_wait_count,
            'average_merchant_wait_time': avg_merchant_wait
        }
    
    def reset(self):
        """重置统计（在新Episode开始时调用）"""
        self.total_rewards = 0.0
        self.step_count = 0
        # v2新增：重置里程碑计数器
        self.completed_count_for_milestone = 0
        # Day 23: 重置商家等餐统计
        self.total_merchant_wait_time = 0.0
        self.merchant_wait_count = 0


class ShapingRewardCalculator(RewardCalculator):
    """
    带Potential-based Reward Shaping的奖励计算器
    
    通过势函数引导Agent更快学习，同时保证最优策略不变
    参考文献：Ng, A. Y., Harada, D., & Russell, S. (1999). 
             Policy invariance under reward transformations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化"""
        super().__init__(config)
        
        self.gamma = self.config.get('discount_factor', 0.99)
        
        logger.info("使用 Potential-based Reward Shaping")
    
    def calculate_step_reward(self,
                             state_before: Dict[str, Any],
                             action: Dict[str, Any],
                             state_after: Dict[str, Any],
                             info: Optional[Dict[str, Any]] = None) -> float:
        """
        计算带Shaping的奖励
        
        Shaped Reward = R(s, a, s') + γ * Φ(s') - Φ(s)
        其中 Φ 是势函数
        
        注意：统计口径修正
        - 父类的calculate_step_reward会累加base_reward到total_rewards
        - 这里额外记录shaping_term，使get_reward_statistics()返回完整的shaped reward
        """
        # 基础奖励（父类会累加到self.total_rewards）
        base_reward = super().calculate_step_reward(state_before, action, state_after, info)
        
        # 势函数项
        potential_before = self._potential_function(state_before)
        potential_after = self._potential_function(state_after)
        shaping_term = self.gamma * potential_after - potential_before
        
        # 统计口径修正：将shaping_term也累加到total_rewards
        # 这样get_reward_statistics()返回的是Agent实际看到的shaped reward
        self.total_rewards += shaping_term
        
        return base_reward + shaping_term
    
    def _potential_function(self, state: Dict[str, Any]) -> float:
        """
        势函数设计
        
        目标：鼓励Agent向"理想状态"靠近
        - 待分配订单少
        - 超时风险低
        - 骑手负载均衡
        
        Args:
            state: 环境状态
        
        Returns:
            势函数值
        """
        # 待分配订单数（越少越好）
        pending_count = len(state.get('pending_orders', []))
        pending_penalty = -pending_count * 10.0
        
        # 系统超时风险（越低越好）
        timeout_risk = self._calculate_system_timeout_risk(state)
        risk_penalty = -timeout_risk * 50.0
        
        # 负载平衡（越高越好）
        load_balance = self._calculate_load_balance(state)
        balance_bonus = load_balance * 20.0
        
        return pending_penalty + risk_penalty + balance_bonus


def test_reward_calculator():
    """测试奖励计算器"""
    print("="*60)
    print("测试 RewardCalculator")
    print("="*60)
    
    # 创建奖励计算器
    calculator = RewardCalculator({
        'reward_type': 'dense',
        'weight_timeout_penalty': 10.0,
        'weight_distance_cost': 0.001
    })
    
    print(f"\n奖励函数配置:")
    print(f"  类型: {calculator.reward_type}")
    print(f"  超时惩罚权重: {calculator.weight_timeout_penalty}")
    print(f"  距离成本权重: {calculator.weight_distance_cost}")
    
    # 模拟状态
    from collections import namedtuple
    Order = namedtuple('Order', ['latest_delivery_time'])
    Courier = namedtuple('Courier', ['current_route'])
    
    state_before = {
        'current_time': 3600.0,
        'pending_orders': [
            Order(5400.0),
            Order(5500.0),
        ],
        'couriers': {
            1: Courier([]),
            2: Courier([('pickup', 1, 123)]),
        }
    }
    
    state_after = {
        'current_time': 3610.0,
        'pending_orders': [
            Order(5500.0),  # 一个订单被分配了
        ],
        'couriers': {
            1: Courier([('pickup', 2, 124)]),  # 分配给骑手1
            2: Courier([('pickup', 1, 123)]),
        }
    }
    
    action = {
        'distance_increase': 1500.0,  # 增加1.5公里
        'action_type': 'assign'
    }
    
    info = {
        'completed_orders': [],
        'delay_justified': False
    }
    
    # 计算奖励
    reward = calculator.calculate_step_reward(state_before, action, state_after, info)
    print(f"\n计算的奖励值: {reward:.4f}")
    
    # 模拟订单完成
    info_with_completion = {
        'completed_orders': [
            {'is_timeout': False, 'slack_time': 120.0}
        ]
    }
    
    reward_with_completion = calculator.calculate_step_reward(
        state_before, action, state_after, info_with_completion
    )
    print(f"带订单完成的奖励: {reward_with_completion:.4f}")
    
    # 测试Shaping版本
    print("\n" + "="*60)
    print("测试 ShapingRewardCalculator")
    print("="*60)
    
    shaping_calc = ShapingRewardCalculator({
        'reward_type': 'dense',
        'discount_factor': 0.99
    })
    
    shaped_reward = shaping_calc.calculate_step_reward(state_before, action, state_after, info)
    print(f"Shaped奖励值: {shaped_reward:.4f}")
    print(f"Shaping增量: {shaped_reward - reward:.4f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_reward_calculator()
