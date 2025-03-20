import glob
import os

import pandas as pd

def main():
    # Get all files in the current directory
    files = glob.glob(os.path.join('datasets/venom', '*.csv'))

    all_df = pd.DataFrame(columns=['文件id', '命名', '描述', '开始时间', '结束时间'])
    for file in files:
        df = pd.read_csv(file)
        # Caption the file id
        df['文件id'] = file.split('/')[-1].split('.')[0].capitalize()
        all_df = pd.concat([all_df, df])

    # Filter out rows where the 'label' column is 'henshin'
    all_df = all_df[all_df['描述'].str.contains('变身')]

    # Save the filtered dataframe to a new csv file
    all_df.to_csv('datasets/venom/henshin_videos.csv', index=False)

if __name__ == "__main__":
    main()