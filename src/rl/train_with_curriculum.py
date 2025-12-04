"""
Day 14: 课程学习训练脚本

专用于执行课程学习训练的入口脚本。
从低负载到高负载逐步训练RL Agent。

使用方法:
    python -m src.rl.train_with_curriculum --config config/rl_config.yaml
    python -m src.rl.train_with_curriculum --quick  # 快速测试模式
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RL训练库
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.error("Stable-Baselines3未安装，请运行: pip install stable-baselines3[extra]")

from .curriculum_learning import (
    CurriculumManager, 
    CurriculumStage,
    CurriculumLearningCallback,
    print_curriculum_overview,
    AdvanceCondition
)
from .rl_environment import DeliveryRLEnvironment
from .train_rl_agent import TrainingMonitorCallback


class CurriculumTrainer:
    """
    课程学习训练器
    
    管理整个课程学习训练流程
    """
    
    def __init__(self, 
                 config_path: str,
                 output_dir: Optional[str] = None,
                 quick_mode: bool = False):
        """
        初始化课程学习训练器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录（可选）
            quick_mode: 快速测试模式
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.quick_mode = quick_mode
        
        # 配置提取
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        self.training_config = self.rl_config.get('training', {})
        
        # 输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(self.rl_config.get('model_save_path', './outputs/rl_training'))
            self.output_dir = base_dir / f"curriculum_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard日志目录
        self.tensorboard_dir = self.output_dir / 'tensorboard'
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建课程管理器
        self.curriculum = self._create_curriculum_manager()
        
        # 保存配置
        self._save_config()
        
        logger.info(f"课程学习训练器初始化完成")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  快速模式: {self.quick_mode}")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _save_config(self) -> None:
        """保存使用的配置"""
        config_save_path = self.output_dir / 'config_used.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def _create_curriculum_manager(self) -> CurriculumManager:
        """创建课程管理器"""
        curriculum_config = self.training_config.get('curriculum', {})
        
        if self.quick_mode:
            # 快速测试模式：简化的课程
            stages = [
                CurriculumStage(
                    name="quick_warmup",
                    description="快速测试-热身",
                    total_orders=100, num_couriers=10,
                    timesteps=2000,
                    advance_condition=AdvanceCondition.FIXED_STEPS,
                    difficulty_score=0.2
                ),
                CurriculumStage(
                    name="quick_main",
                    description="快速测试-主训练",
                    total_orders=200, num_couriers=10,
                    timesteps=3000,
                    advance_condition=AdvanceCondition.FIXED_STEPS,
                    difficulty_score=0.5
                )
            ]
            return CurriculumManager(stages=stages, config=curriculum_config)
        else:
            # 使用默认的7阶段课程
            return CurriculumManager(config=curriculum_config)
    
    def create_env(self, **kwargs) -> DeliveryRLEnvironment:
        """
        创建RL环境
        
        Args:
            **kwargs: 环境配置覆盖参数
            
        Returns:
            RL环境实例
        """
        # 合并配置
        sim_config = {**self.sim_config, **kwargs}
        
        env = DeliveryRLEnvironment(
            simulation_config=sim_config,
            rl_config=self.rl_config
        )
        return env
    
    def train(self) -> None:
        """执行课程学习训练"""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3未安装，无法训练")
        
        logger.info("="*70)
        logger.info("Day 14: 课程学习训练")
        logger.info("="*70)
        
        # 打印课程概览
        print_curriculum_overview(self.curriculum)
        
        # 创建初始模型
        model = None
        
        # 逐阶段训练
        while not self.curriculum.is_completed:
            stage = self.curriculum.current_stage
            stage_idx = self.curriculum.current_stage_idx
            
            logger.info(f"\n{'='*60}")
            logger.info(f"开始阶段 {stage_idx + 1}/{len(self.curriculum.stages)}: {stage.name}")
            logger.info(f"{'='*60}")
            logger.info(f"  描述: {stage.description}")
            logger.info(f"  订单数: {stage.total_orders}")
            logger.info(f"  骑手数: {stage.num_couriers}")
            logger.info(f"  训练步数: {stage.timesteps:,}")
            logger.info(f"  难度评分: {stage.difficulty_score:.2f}")
            
            # 获取阶段配置并创建环境
            stage_config = self.curriculum.get_stage_config()
            env = DummyVecEnv([lambda: self.create_env(**stage_config)])
            
            # 创建或更新模型
            if model is None:
                model = self._create_model(env)
                # 配置TensorBoard
                new_logger = configure(str(self.tensorboard_dir), ["stdout", "tensorboard"])
                model.set_logger(new_logger)
            else:
                model.set_env(env)
            
            # 创建回调
            callbacks = self._create_callbacks(stage)
            
            # 记录阶段开始时间
            stage_start_time = time.time()
            
            # 训练该阶段
            try:
                model.learn(
                    total_timesteps=stage.timesteps,
                    callback=callbacks,
                    reset_num_timesteps=False,
                    progress_bar=True
                )
                
                # 记录阶段耗时
                stage_duration = time.time() - stage_start_time
                logger.info(f"阶段 {stage.name} 完成，耗时: {stage_duration/60:.1f}分钟")
                
                # 保存阶段模型
                stage_model_path = self.output_dir / f"stage_{stage_idx + 1}_{stage.name}"
                model.save(stage_model_path)
                logger.info(f"阶段模型已保存: {stage_model_path}")
                
                # 保存课程进度
                progress_path = self.output_dir / "curriculum_progress.json"
                self.curriculum.save_progress(progress_path)
                
            except KeyboardInterrupt:
                logger.info(f"\n训练在阶段 {stage.name} 被中断")
                interrupt_path = self.output_dir / f"interrupted_stage_{stage_idx + 1}"
                model.save(interrupt_path)
                self.curriculum.save_progress(self.output_dir / "curriculum_progress.json")
                break
            
            # 检查是否应该进阶
            # 在固定步数模式下，直接进阶
            if stage.advance_condition == AdvanceCondition.FIXED_STEPS:
                self.curriculum.advance_stage()
            # 在性能阈值模式下，检查性能
            elif self.curriculum.should_advance():
                self.curriculum.advance_stage()
            else:
                # 未达到阈值，但已完成步数，强制进阶
                logger.warning(f"阶段 {stage.name} 未达到性能阈值，但已完成训练步数，强制进阶")
                self.curriculum.advance_stage()
        
        # 保存最终模型
        final_model_path = self.output_dir / "final_curriculum_model"
        model.save(final_model_path)
        logger.info(f"\n课程学习完成！最终模型: {final_model_path}")
        
        # 保存最终摘要
        self._save_training_summary()
        
        # 打印TensorBoard命令
        print(f"\nTensorBoard命令: tensorboard --logdir={self.tensorboard_dir}")
    
    def _create_model(self, env) -> PPO:
        """创建PPO模型"""
        ppo_config = self.training_config.get('ppo', {})
        policy_config = self.training_config.get('policy', {})
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.training_config.get('learning_rate', 3e-4),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=self.training_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.01),
            policy_kwargs=dict(
                net_arch=policy_config.get('net_arch', [256, 256])
            ),
            verbose=1,
            seed=self.config.get('seed', 42)
        )
        return model
    
    def _create_callbacks(self, stage: CurriculumStage) -> CallbackList:
        """创建训练回调"""
        callbacks = []
        
        # 训练监控回调
        monitor_callback = TrainingMonitorCallback(
            check_freq=500,
            log_dir=str(self.output_dir),
            early_stop_patience=100,
            min_improvement=0.1,
            verbose=1
        )
        callbacks.append(monitor_callback)
        
        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=max(stage.timesteps // 5, 1000),  # 每阶段保存5次
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix=f"curriculum_{stage.name}"
        )
        callbacks.append(checkpoint_callback)
        
        return CallbackList(callbacks)
    
    def _save_training_summary(self) -> None:
        """保存训练摘要"""
        summary = self.curriculum.get_curriculum_summary()
        summary['training_completed_at'] = datetime.now().isoformat()
        summary['output_dir'] = str(self.output_dir)
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练摘要已保存: {summary_path}")
        
        # 打印最终摘要
        print("\n" + "="*70)
        print("课程学习训练摘要")
        print("="*70)
        print(f"总训练步数: {summary['total_timesteps_trained']:,}")
        print(f"完成阶段数: {summary['current_stage_idx']}/{summary['total_stages']}")
        print(f"训练进度: {summary['progress_percentage']:.1f}%")
        print("\n各阶段性能:")
        for name, perf in summary['performances'].items():
            if perf['episodes_completed'] > 0:
                print(f"  {name}: 完成率={perf['completion_rate']:.1%}, "
                      f"超时率={perf['timeout_rate']:.1%}, "
                      f"平均奖励={perf['avg_reward']:.2f}")
        print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Day 14: 课程学习训练 - 从低负载到高负载逐步训练'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/rl_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录（可选）'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速测试模式（简化课程）'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练（指定progress.json路径）'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Day 14: 课程学习训练")
    print("="*70)
    print(f"配置文件: {args.config}")
    print(f"快速模式: {args.quick}")
    print(f"输出目录: {args.output or '自动生成'}")
    
    # 创建训练器
    trainer = CurriculumTrainer(
        config_path=args.config,
        output_dir=args.output,
        quick_mode=args.quick
    )
    
    # 恢复进度（如果指定）
    if args.resume:
        if trainer.curriculum.load_progress(Path(args.resume)):
            print(f"已从 {args.resume} 恢复进度")
    
    # 开始训练
    trainer.train()
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()
