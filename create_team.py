import pandas as pd
import numpy as np
from pulp import *

def optimize_team(file_path):
    df = pd.read_excel(file_path)
    
    print("Available columns in the Excel file:")
    print(df.columns.tolist())
    
    # Ensure required columns exist
    required_columns = ['name', 'pos', 'money', 'pgp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the Excel file.")
    
    # Try to find the point projection column
    point_proj_col = next((col for col in df.columns if 'point' in col.lower() and 'projection' in col.lower()), None)
    
    if point_proj_col:
        print(f"Using '{point_proj_col}' as the point projection column.")
        # Calculate points per game, using 0.9 * pgp when point projection is null
        df['ppg'] = np.where(df[point_proj_col].isnull(),
                             0.9 * df['pgp'],
                             df[point_proj_col] / 82)
    else:
        print("Point projection column not found. Using 0.9 * pgp for ppg calculation.")
        df['ppg'] = 0.9 * df['pgp']
    
    # Calculate the objective value
    df['objective'] = 0 * df['pgp'] + 1 * df['ppg']

    prob = LpProblem("Team Optimization", LpMaximize)
    player_vars = LpVariable.dicts("Players", df.index, cat='Binary')
    
    # Update the objective function
    prob += lpSum([player_vars[i] * df.loc[i, 'objective'] for i in df.index])
    
    # Constraints remain the same
    prob += lpSum([player_vars[i] * df.loc[i, 'money'] for i in df.index]) <= 82
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'pos'] in ['L', 'C', 'R']]) == 13
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'pos'] == 'D']) == 8
    
    prob.solve()
    
    selected_players = [i for i in df.index if player_vars[i].value() == 1]
    selected_df = df.loc[selected_players]
    
    return format_output(selected_df)

def format_output(df):
    forwards = df[df['pos'].isin(['L', 'C', 'R'])].sort_values('objective', ascending=False)
    defensemen = df[df['pos'] == 'D'].sort_values('objective', ascending=False)
    
    def format_player(player):
        return (f"{player['name']} ({player['pos']}): "
                f"money ${player['money']:.2f}, "
                f"pgp: {player['pgp']:.2f}, "
                f"ppg: {player['ppg']:.2f}, "
                f"objective: {player['objective']:.2f}")

    output = "Forwards:\n"
    output += "\n".join(format_player(player) for _, player in forwards.iterrows())
    
    output += "\n\nDefensemen:\n"
    output += "\n".join(format_player(player) for _, player in defensemen.iterrows())
    
    output += f"\n\nTotal Objective Value: {df['objective'].sum():.2f}"
    output += f"\nTotal PGP: {df['pgp'].sum():.2f}"
    output += f"\nTotal PPG: {df['ppg'].sum():.2f}"
    output += f"\nTotal Cost: ${df['money'].sum():.2f} million"
    
    return output

file_path = 'players.xlsx'
result = optimize_team(file_path)
print(result)