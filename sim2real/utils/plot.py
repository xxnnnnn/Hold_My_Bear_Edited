"""
Utility script to plot joint data from saved npz files.
Usage: python -m sim2real.utils.plot <npz_file_path>
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_joint_data(npz_path):
    """Load joint data from npz file."""
    # Load with allow_pickle=True to support object arrays (joint_names, record_range)
    data = np.load(npz_path, allow_pickle=True)
    
    # Load metadata
    joint_names = []
    if 'joint_names' in data:
        joint_names = data['joint_names'].tolist()
    record_range = None
    if 'record_range' in data:
        record_range = data['record_range'].item() if data['record_range'].size > 0 else None
    
    # Reconstruct wrist_data dictionary structure (for backward compatibility)
    wrist_data = {
        'left': {
            'roll': {},
            'pitch': {},
            'yaw': {}
        },
        'right': {
            'roll': {},
            'pitch': {},
            'yaw': {}
        }
    }
    
    for side in ['left', 'right']:
        for joint_type in ['roll', 'pitch', 'yaw']:
            for data_type in ['cmd_pos', 'actual_pos', 'kp']:
                key = f"{side}_{joint_type}_{data_type}"
                if key in data:
                    wrist_data[side][joint_type][data_type] = data[key]
                else:
                    wrist_data[side][joint_type][data_type] = np.array([])
    
    # Load tracked joint data
    joint_data = {}
    for joint_name in joint_names:
        safe_name = joint_name.replace(' ', '_').replace('-', '_')
        joint_data[joint_name] = {
            'cmd_pos': data.get(f"{safe_name}_cmd_pos", np.array([])),
            'actual_pos': data.get(f"{safe_name}_actual_pos", np.array([])),
            'kp': data.get(f"{safe_name}_kp", np.array([]))
        }
    
    return wrist_data, joint_data, record_range


def plot_wrist_data(wrist_data, output_dir=None, output_prefix=None):
    """Plot wrist data and save to files."""
    # Determine output directory and prefix
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_prefix is None:
        output_prefix = "wrist_plots"
    
    # Create two figures: one for left wrist, one for right wrist
    for side in ['left', 'right']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{side.capitalize()} Wrist Data', fontsize=16, fontweight='bold')
        
        joint_types = ['roll', 'pitch', 'yaw']
        colors = ['blue', 'green', 'red']
        
        # First row: position plots (cmd_pos and actual_pos)
        for col, (joint_type, color) in enumerate(zip(joint_types, colors)):
            ax = axes[0, col]
            data = wrist_data[side][joint_type]
            
            cmd_pos = data.get('cmd_pos', np.array([]))
            actual_pos = data.get('actual_pos', np.array([]))
            
            cmd_pos_len = len(cmd_pos)
            actual_pos_len = len(actual_pos)
            
            if cmd_pos_len > 0 or actual_pos_len > 0:
                # Use minimum length to ensure arrays match
                min_len = min(cmd_pos_len, actual_pos_len) if cmd_pos_len > 0 and actual_pos_len > 0 else max(cmd_pos_len, actual_pos_len)
                
                if min_len > 0:
                    steps = np.arange(min_len)
                    if cmd_pos_len >= min_len:
                        ax.plot(steps, cmd_pos[:min_len], label='Command Position', color=color, linestyle='-', linewidth=1.5)
                    if actual_pos_len >= min_len:
                        ax.plot(steps, actual_pos[:min_len], label='Actual Position', color=color, linestyle='--', linewidth=1.5)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Position (rad)')
                    ax.set_title(f'{joint_type.capitalize()} Position')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{joint_type.capitalize()} Position')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{joint_type.capitalize()} Position')
        
        # Second row: kp plots
        for col, (joint_type, color) in enumerate(zip(joint_types, colors)):
            ax = axes[1, col]
            data = wrist_data[side][joint_type]
            
            kp = data.get('kp', np.array([]))
            
            if len(kp) > 0:
                steps = np.arange(len(kp))
                ax.plot(steps, kp, label='KP', color=color, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel('KP')
                ax.set_title(f'{joint_type.capitalize()} KP')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{joint_type.capitalize()} KP')
        
        plt.tight_layout()
        
        # Save figure
        plot_filename = output_dir / f"{output_prefix}_{side}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Wrist plot saved to {plot_filename}")
        
        plt.close(fig)  # Close figure to free memory


def plot_joint_data(joint_data, record_range=None, output_dir=None, output_prefix=None, group_by="body_part", separate_kp=False):
    """Plot joint data and save to files."""
    # Determine output directory and prefix
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_prefix is None:
        output_prefix = "joint_plots"
    
    if not joint_data:
        print("No joint data to plot")
        return
    
    # Group joints by body part if requested
    if group_by == "body_part":
        upper_body_joints = []
        lower_body_joints = []
        
        for joint_name in joint_data.keys():
            if any(keyword in joint_name.lower() for keyword in ['shoulder', 'elbow', 'wrist']):
                upper_body_joints.append(joint_name)
            else:
                lower_body_joints.append(joint_name)
        
        groups = []
        if upper_body_joints:
            groups.append(('upper_body', upper_body_joints))
        if lower_body_joints:
            groups.append(('lower_body', lower_body_joints))
    else:
        # Plot all joints together
        groups = [('all', list(joint_data.keys()))]
    
    # Plot each group
    for group_name, joint_names in groups:
        num_joints = len(joint_names)
        if num_joints == 0:
            continue
        
        # Determine subplot layout
        if separate_kp:
            # Two rows: positions (cmd+actual in one plot) and kp
            ncols = min(4, num_joints)
            nrows = (num_joints + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 5 * nrows))
            if nrows * ncols == 1:
                axes = np.array([[axes, axes], [axes, axes]])
            elif nrows == 1:
                axes = axes.reshape(2, -1) if axes.ndim == 1 else axes
            fig.suptitle(f'{group_name.replace("_", " ").title()} Joint Data', fontsize=16, fontweight='bold')
        else:
            # One row per joint, two columns: position (cmd+actual) and kp
            ncols = 2
            nrows = num_joints
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1) if axes.ndim == 1 else axes
            fig.suptitle(f'{group_name.replace("_", " ").title()} Joint Data', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
        
        for idx, joint_name in enumerate(joint_names):
            data = joint_data[joint_name]
            color = colors[idx]
            
            cmd_pos = data.get('cmd_pos', np.array([]))
            actual_pos = data.get('actual_pos', np.array([]))
            kp = data.get('kp', np.array([]))
            
            # Short joint name for display
            short_name = joint_name.replace('_joint', '').replace('_', ' ').title()
            
            if separate_kp:
                # Position plot
                row = (idx // ncols) * 2
                col = idx % ncols
                ax = axes[row, col]
                
                cmd_pos_len = len(cmd_pos)
                actual_pos_len = len(actual_pos)
                
                if cmd_pos_len > 0 or actual_pos_len > 0:
                    min_len = min(cmd_pos_len, actual_pos_len) if cmd_pos_len > 0 and actual_pos_len > 0 else max(cmd_pos_len, actual_pos_len)
                    if min_len > 0:
                        steps = np.arange(min_len)
                        if cmd_pos_len >= min_len:
                            ax.plot(steps, cmd_pos[:min_len], label='Command', color=color, linestyle='-', linewidth=1.5)
                        if actual_pos_len >= min_len:
                            ax.plot(steps, actual_pos[:min_len], label='Actual', color=color, linestyle='--', linewidth=1.5)
                        ax.set_xlabel('Step')
                        ax.set_ylabel('Position (rad)')
                        ax.set_title(short_name)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # KP plot
                ax = axes[row + 1, col]
                if len(kp) > 0:
                    steps = np.arange(len(kp))
                    ax.plot(steps, kp, label='KP', color=color, linewidth=1.5)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('KP')
                    ax.set_title(f'{short_name} KP')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                # Combined plot: position (cmd & actual) and kp
                # Position plot
                ax = axes[idx, 0]
                cmd_pos_len = len(cmd_pos)
                actual_pos_len = len(actual_pos)
                if cmd_pos_len > 0 or actual_pos_len > 0:
                    min_len = min(cmd_pos_len, actual_pos_len) if cmd_pos_len > 0 and actual_pos_len > 0 else max(cmd_pos_len, actual_pos_len)
                    if min_len > 0:
                        steps = np.arange(min_len)
                        if cmd_pos_len >= min_len:
                            ax.plot(steps, cmd_pos[:min_len], label='Command', color=color, linestyle='-', linewidth=1.5)
                        if actual_pos_len >= min_len:
                            ax.plot(steps, actual_pos[:min_len], label='Actual', color=color, linestyle='--', linewidth=1.5)
                        ax.set_xlabel('Step')
                        ax.set_ylabel('Position (rad)')
                        ax.set_title(short_name)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                # KP plot
                ax = axes[idx, 1]
                if len(kp) > 0:
                    steps = np.arange(len(kp))
                    ax.plot(steps, kp, label='KP', color=color, linewidth=1.5)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('KP')
                    ax.set_title(f'{short_name} KP')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_filename = output_dir / f"{output_prefix}_{group_name}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Joint plot saved to {plot_filename}")
        
        plt.close(fig)  # Close figure to free memory


def main():
    parser = argparse.ArgumentParser(description="Plot joint data from npz file")
    parser.add_argument("npz_file", type=str, help="Path to the npz file containing joint data")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory for plots (default: same directory as npz file)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output filename prefix (default: joint_plots)")
    parser.add_argument("--group_by", type=str, default="body_part",
                        choices=["body_part", "none"],
                        help="Group joints by body part (upper/lower) or plot all together")
    parser.add_argument("--separate_kp", action="store_true",
                        help="Separate KP plots from position plots")
    
    args = parser.parse_args()
    
    # Load data
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        print(f"Error: File {npz_path} does not exist")
        return
    
    print(f"Loading joint data from {npz_path}")
    wrist_data, joint_data, record_range = load_joint_data(npz_path)
    
    if record_range:
        print(f"Record range: {record_range}")
    
    # Determine output directory (default to same directory as npz file)
    if args.output_dir is None:
        output_dir = npz_path.parent
    else:
        output_dir = Path(args.output_dir)
    
    # Determine output prefix (default to joint_plots)
    if args.output_prefix is None:
        output_prefix = "joint_plots"
    else:
        output_prefix = args.output_prefix
    
    # Plot wrist data (for backward compatibility)
    has_wrist_data = any(
        len(wrist_data[side][joint_type].get('cmd_pos', [])) > 0
        for side in ['left', 'right']
        for joint_type in ['roll', 'pitch', 'yaw']
    )
    if has_wrist_data:
        plot_wrist_data(wrist_data, output_dir=output_dir, output_prefix="wrist_plots")
    
    # Plot joint data
    if joint_data:
        plot_joint_data(joint_data, record_range=record_range, 
                       output_dir=output_dir, output_prefix=output_prefix,
                       group_by=args.group_by, separate_kp=args.separate_kp)
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()

