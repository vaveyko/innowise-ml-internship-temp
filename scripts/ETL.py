import pandas as pd
from pandas import DataFrame
from rapidfuzz import fuzz, process


def find_equals_name(strings: list, threshold: int = 80) -> dict:
    """
    Find and map similar strings from a list using fuzzy matching.
    If similarity more then the threshold, prompts user to
    choose which string is correct. Returns a mapping of duplicates.

    :param strings: list of strings
    :param threshold: from 0 to 100
    :return: dictionary {incorrect_id: correct_id}
    """

    results = []
    equal_shop_name: dict = {}
    for i, string in enumerate(strings):
        choices = strings[i + 1 :]

        match = process.extractOne(string, choices, scorer=fuzz.ratio)

        if match and match[1] >= threshold:
            results.append(
                {
                    "original_1": string,
                    "original_2": choices[match[2]],
                    "similarity": match[1],
                    "id_1": i,
                    "id_2": match[2] + i + 1,
                }
            )
    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, res in enumerate(results):
        print(
            f"({i + 1}/{len(results)})"
            f"Which name is correct (similarity {res['similarity']: .5})\n"
            f"\t1) {res['original_1']} | index={res['id_1']}\n"
            f"\t2) {res['original_2']} | index={res['id_2']}\n"
            f"\t3) Skip\n"
            f"\t4) Skip all"
        )
        choice = int(input())
        match choice:
            case 1:
                equal_shop_name[res["id_2"]] = res["id_1"]
            case 2:
                equal_shop_name[res["id_1"]] = res["id_2"]
            case 3:
                pass
            case 4:
                break
    return equal_shop_name


def delete_equal_shop_name(
    sales_df: DataFrame, shop_df: DataFrame, test_df: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Delete from shop_df equal shop names, reset id and replace old shop_id with new ones
    in sales_df and test_df

    :param sales_df:
    :param shop_df:
    :param test_df:
    :return: tuple of changed dataframes
    """
    # "Жуковский ул. Чкалова 39м?" and "Жуковский ул. Чкалова 39м²"
    # (correct_id=11, incorrect=10)
    # "Якутск Орджоникидзе, 56" and "!Якутск Орджоникидзе, 56 фран""
    # (correct_id=57, incorrect=0)
    # "Якутск ТЦ "Центральный"" and "!Якутск ТЦ "Центральный" фран"
    # (correct_id=58, incorrect=1)

    # equal_shop_map = {
    #     10: 11,
    #     0: 57,
    #     1: 58,
    # }
    equal_shop_map = find_equals_name(shop_df["shop_name"].to_list())
    shop_df = shop_df.copy()
    shop_df["new_shop_id"] = shop_df["shop_id"]

    # replace equal
    shop_df["new_shop_id"] = shop_df["new_shop_id"].replace(equal_shop_map)
    sales_df["shop_id"] = sales_df["shop_id"].replace(equal_shop_map)
    test_df["shop_id"] = test_df["shop_id"].replace(equal_shop_map)

    # leave only one shop of each pair of equals
    shop_df = shop_df[shop_df["shop_id"] == shop_df["new_shop_id"]]

    # drop help column and reset index
    shop_df = shop_df.drop(["new_shop_id"], axis=1)
    shop_df = shop_df.reset_index(drop=True)

    old_new_id_map = dict(zip(shop_df["shop_id"], shop_df.index))

    # replace old ids with new after .reset_index
    sales_df["shop_id"] = sales_df["shop_id"].replace(old_new_id_map)
    test_df["shop_id"] = test_df["shop_id"].replace(old_new_id_map)
    shop_df["shop_id"] = shop_df.index

    return sales_df, shop_df, test_df


def remove_outliers(sales_df: DataFrame) -> DataFrame:
    """
    Removes outliers of count of sales a month using MAD method

    :param sales_df: dataframe with "item_cnt_month" column
    :return: modified sales_df without outliers
    """

    # TODO check other ways to handle with outliers when start trying models
    # calculate MAD only on data != 1 because 1 is 60% of data
    # and MAD became 0
    sales_without_ones = sales_df[sales_df["item_cnt_month"] != 1]
    mean = sales_without_ones["item_cnt_month"].mean()
    MAD = (sales_without_ones["item_cnt_month"] - mean).abs().mean() * 1.4826

    sales_df = sales_df[
        sales_df["item_cnt_month"].between(mean - 3 * MAD, mean + 3 * MAD)
    ]

    print(
        f"MAD border: [{mean - 3 * MAD}, {mean + 3 * MAD}]\
    Mean: {mean}\
    MAD: {MAD}"
    )
    return sales_df


def from_day_to_month(sales_df: DataFrame) -> DataFrame:
    """
    Function aggregates daily sales by month, shop id, item id and modifies
    DataFrame for future using

    :param sales_df:
    :return: modified DataFrame
    """

    sales_df["date"] = pd.to_datetime(sales_df["date"], dayfirst=True)
    sales_df["month"] = sales_df["date"].dt.month

    sales_df = (
        sales_df.groupby(["date_block_num", "shop_id", "item_id"])
        .agg(
            {
                "item_cnt_day": "sum",
                "item_price": "mean",
                "item_category_id": "first",
                "month": "first",
            }
        )
        .reset_index()
    )
    # TODO check if item_price is useful column
    sales_df = sales_df.drop("item_price", axis=1)
    sales_df = sales_df.rename(columns={"item_cnt_day": "item_cnt_month"})

    return sales_df


def fix_data(
    sales_df: DataFrame,
    items_df: DataFrame,
    shop_df: DataFrame,
    test_df: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Preprocesses dataframes. Copes with outliers. Bring into the correct form.

    :param sales_df:
    :param items_df:
    :param shop_df:
    :param test_df:
    :return: tuple of preprocessed DataFrames
    """

    # Check negative price
    negative_mask = sales_df["item_price"] < 0
    negative_count = negative_mask.sum()
    if negative_count > 0:
        print(f"Negative prices found ({negative_count})")
        sales_df = sales_df[~negative_mask]

    # equal shop names
    sales_df, shop_df, test_df = delete_equal_shop_name(sales_df, shop_df, test_df)

    # Add category id in data
    sales_df = sales_df.join(items_df["item_category_id"], on="item_id")
    test_df = test_df.join(items_df["item_category_id"], on="item_id")

    # Move to cnt_month
    sales_df = from_day_to_month(sales_df)

    # Remove outliers
    sales_df = remove_outliers(sales_df)

    return sales_df, shop_df, test_df


def main(data_dir: str = "../data"):
    """
    Read data, fix it and load correct version into memory

    :return: None
    """
    # read data
    items_df = pd.read_csv(data_dir + "/items.csv")
    sales_train_df = pd.read_csv(data_dir + "/sales_train.csv")
    shops_df = pd.read_csv(data_dir + "/shops.csv")
    test_df = pd.read_csv(data_dir + "/test.csv")

    # fix
    sales_train_df_fix, shops_df_fix, test_df_fix = fix_data(
        sales_train_df,
        items_df,
        shops_df,
        test_df,
    )

    # convert to calculate percent of deleted data
    sales_train_df_old = from_day_to_month(
        sales_train_df.join(items_df["item_category_id"], on="item_id")
    )
    percent = 1 - len(sales_train_df_fix) / len(sales_train_df_old)
    print(f"Was deleted {percent: .2%} of data")

    sales_train_df_fix.to_csv(
        data_dir + "/preprocessed/sales_train_preprocessed.csv", index=False
    )
    shops_df_fix.to_csv(data_dir + "/preprocessed/shops_preprocessed.csv", index=False)
    test_df_fix.to_csv(data_dir + "/preprocessed/test_preprocessed.csv", index=False)
    return


if __name__ == "__main__":
    main()
