import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from scipy import signal
from scipy.stats import zscore

def simple_plot(data, start_time=10, end_time=20, figsize=(15, 12), spaceing=100):
    """
    지정된 시간 범위의 EEG 데이터를 subplot으로 시각화
    
    Parameters:
    -----------
    data : pandas.DataFrame
        EEG 데이터 (time 컬럼과 pin 컬럼들 포함)
    start_time : float
        시작 시간 (초)
    end_time : float  
        종료 시간 (초)
    figsize : tuple
        그래프 크기
    """
    # 시간 범위로 데이터 필터링
    time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    filtered_data = data[time_mask].copy()
    
    if len(filtered_data) == 0:
        print(f"Warning: No data found in time range {start_time}-{end_time} seconds")
        return
    
    # pin 컬럼들만 추출 (time 제외)
    pin_columns = [col for col in data.columns if col != 'time']
    n_channels = len(pin_columns)
    
    # subplot 레이아웃 계산 (세로로 배치)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    
    # 채널이 1개인 경우 axes를 리스트로 만들기
    if n_channels == 1:
        axes = [axes]
    
    # 모든 채널의 데이터 범위 계산 (동일한 y축 범위 적용)
    all_data = []
    for pin_name in pin_columns:
        all_data.extend(filtered_data[pin_name].values)
    
    all_data = np.array(all_data)
    global_max = np.max(all_data)
    global_min = np.min(all_data)
    
    y_min = np.floor(global_min / 100) * 100  # global_min 보다 작은 값중 50의 배수
    y_max = np.ceil(global_max / 100) * 100  # global_max 보다 큰 값중 50의 배수
    
    # 각 채널별로 subplot 그리기
    for i, pin_name in enumerate(pin_columns):
        axes[i].plot(filtered_data['time'], filtered_data[pin_name], 
                    linewidth=0.8, color='black')
        
        axes[i].set_ylabel(f'{pin_name}\n(µV)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'Single Spike Signal - {pin_name}', fontsize=11, pad=5)
        
        # 모든 채널에 동일한 y축 범위 적용
        axes[i].set_ylim(y_min, y_max)
        # y축 틱을 100µV 간격으로 설정 (100의 배수만)
        tick_start = y_min
        tick_end = y_max
        y_ticks = np.arange(tick_start, tick_end, spaceing)
        axes[i].set_yticks(y_ticks)
        # x축 범위를 정확히 맞춤
        axes[i].set_xlim(start_time, end_time)
    
    # x축 레이블은 마지막 subplot에만
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    # 전체 제목
    fig.suptitle(f'Single Spike Signals ({start_time}-{end_time} seconds)', 
                fontsize=14, fontweight='bold')
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # 통계 정보 출력
    print(f"📊 Data Range Summary ({start_time}-{end_time}s):")
    print(f"   - Total samples: {len(filtered_data):,}")
    print(f"   - Sampling rate: ~{len(filtered_data)/(end_time-start_time):.0f} Hz")
    print(f"   - Channels: {pin_columns}")