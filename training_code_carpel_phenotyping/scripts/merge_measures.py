import click
import pandas as pd


@click.command()
@click.argument('manual_fpath')
@click.argument('auto_fpath')
def main(manual_fpath, auto_fpath):

    df_manual = pd.read_csv(manual_fpath, index_col=0)

    df_auto = pd.read_csv(auto_fpath, index_col=0)

    print(df_manual.join(df_auto).to_csv())


if __name__ == "__main__":
    main()