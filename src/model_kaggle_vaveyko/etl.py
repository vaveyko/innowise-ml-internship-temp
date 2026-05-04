import pandas as pd
from pandas import DataFrame
from rapidfuzz import fuzz, process
from sklearn.preprocessing import LabelEncoder


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
    sales_df: DataFrame,
    shop_df: DataFrame,
    test_df: DataFrame,
    is_human_in_loop: bool = True,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Delete from shop_df equal shop names, reset id and replace old shop_id with new ones
    in sales_df and test_df

    :param is_human_in_loop:
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

    if is_human_in_loop:
        equal_shop_map = find_equals_name(shop_df["shop_name"].to_list())
    else:
        equal_shop_map = {
            10: 11,
            0: 57,
            1: 58,
        }
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

    # calculate MAD only on data != 1 because 1 is 60% of data
    # and MAD became 0
    median = sales_df["item_cnt_month"].median()
    sales_without_ones = sales_df[sales_df["item_cnt_month"] != 1]
    MAD = (sales_without_ones["item_cnt_month"] - median).abs().mean() * 1.4826

    sales_df = sales_df[
        sales_df["item_cnt_month"].between(median - 3 * MAD, median + 3 * MAD)
    ]

    print(
        f"MAD border: [{median - 3 * MAD}, {median + 3 * MAD}]\
    Median: {median}\
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
    sales_df = sales_df.rename(
        columns={"item_cnt_day": "item_cnt_month", "item_price": "mean_month_price"}
    )

    return sales_df


def create_lag(
    df: pd.DataFrame,
    for_column: str,
    by_column: str,
    new_name: str,
    lag_shift: int = 1,
    insert_NAN_col: bool = True,
) -> pd.DataFrame:
    """
    Creates a lagged feature by grouping, averaging, shifting in time,
    and merging back; optionally adds a NaN flag and fills missing with 0.

    :param df:
    :param for_column:
    :param by_column:
    :param new_name:
    :param lag_shift:
    :param insert_NAN_col:
    :return:
    """
    df_grouped = (
        df.groupby(["date_block_num", for_column])[by_column].mean().reset_index()
    )
    df_grouped["date_block_num"] = df_grouped["date_block_num"] + lag_shift
    df_grouped = df_grouped.rename(columns={by_column: new_name})

    df = df.merge(right=df_grouped, how="left", on=["date_block_num", for_column])
    if insert_NAN_col:
        df[new_name + "_NAN"] = df[new_name].isna().astype(int)
    df[new_name] = df[new_name].fillna(0)
    return df


def create_lags(
    df: pd.DataFrame,
    for_columns: list[str],
    by_column: str,
    new_names: list[str],
    lag_shifts: list[int],
    insert_NAN_col: bool = True,
) -> pd.DataFrame:
    """
    Creates multiple lagged features for different columns and
    shifts by repeatedly applying the lag function.

    :param df:
    :param for_columns:
    :param by_column:
    :param new_names:
    :param lag_shifts:
    :param insert_NAN_col:
    :return:
    """
    for lag_shift in lag_shifts:
        for for_column, new_name in zip(for_columns, new_names):
            df = create_lag(
                df,
                for_column,
                by_column,
                new_name + "_" + str(lag_shift),
                lag_shift,
                insert_NAN_col,
            )
    return df


def label_encode_feat(df: pd.DataFrame, columns: list[str], drop=True) -> pd.DataFrame:
    """
    Encodes categorical columns with label encoding; by default drops original columns.

    :param df:
    :param columns:
    :param drop:
    :return:
    """
    for col in columns:
        df[col + "_encoded"] = LabelEncoder().fit_transform(df[col])
    if drop:
        df = df.drop(columns=columns)
    return df


def make_trend_from_lags(
    df: pd.DataFrame, lag_name: str, new_name: str, lag_pairs: list[tuple[int, int]]
) -> pd.DataFrame:
    """
    Creates trend features as differences between specified lag pairs.

    :param df:
    :param lag_name:
    :param new_name:
    :param lag_pairs:
    :return:
    """
    new_cols = {}

    for first, second in lag_pairs:
        col_name = f"{new_name}_{first}_{second}"
        col_first = f"{lag_name}_{first}"
        col_second = f"{lag_name}_{second}"

        new_cols[col_name] = df[col_first] - df[col_second]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def make_trends(
    df: pd.DataFrame,
    cols: list[str],
    lag_pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Builds trend features for multiple lagged columns using given lag pairs.

    :param df:
    :param cols:
    :param lag_pairs:
    :return:
    """
    for col in cols:
        new_name = col.rsplit("_", 1)[0] + "_trend"
        df = make_trend_from_lags(
            df,
            col,
            new_name,
            lag_pairs,
        )
    return df


def ratio_lag_mean_1_3_to_lag_1(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Computes mean of lags 1–3 and its ratio to lag 1 for given columns,
    then appends as new features.

    :param df:
    :param cols:
    :return:
    """
    new_cols = {}
    for col in cols:
        mean_col = col + "_mean_1_3"
        ratio_col = "ratio_" + col + "_mean_1_3_to_lag1"
        mean_vals = df[[col + "_1", col + "_2", col + "_3"]].mean(axis=1)
        ratio_vals = mean_vals / (df[col + "_1"] + 1e-6)

        new_cols[mean_col] = mean_vals
        new_cols[ratio_col] = ratio_vals
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def make_collab_lag(
    df: DataFrame,
    names: list[list[str]],
    new_names: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Creates lagged collaborative features by grouping columns and
    shifting target values for given lags.

    :param df:
    :param names:
    :param new_names:
    :param lags:
    :return:
    """
    new_cols = {}

    for new_name, group in zip(new_names, names):
        group = df.groupby(group)["item_cnt_month"]
        for lag in lags:
            col_name = f"{new_name}_{lag}"
            feature = group.shift(lag).fillna(0)
            new_cols[col_name] = feature

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def make_single_dataset(sales_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # take sales only with shop_id that containce in test_data
    sales_train_df_uniq = sales_df[
        sales_df["shop_id"].isin(test_df["shop_id"].unique())
    ]

    # make single dataset for future feature extraction
    hole_data = pd.concat([sales_train_df_uniq, test_df])
    hole_data["date_block_num"] = hole_data["date_block_num"].fillna(34)
    hole_data["month"] = hole_data["month"].fillna(11)
    hole_data["ID"] = hole_data["ID"].fillna(-1)

    int_columns = ["date_block_num", "month", "ID"]

    # make some columns as int to reduce memory щмукруфв
    for col in int_columns:
        hole_data[col] = hole_data[col].astype(int)
    return hole_data


def add_shop_features(
    hole_df: pd.DataFrame, shop_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # fix some incorrect names
    shop_df.at[43, "shop_name"] = 'Сергиев_Посад ТЦ "7Я"'
    shop_df.at[17, "shop_name"] = 'Москва Магазин "Распродажа"'
    shop_df.at[7, "shop_name"] = "- Выездная_Торговля"
    shop_df.at[4, "shop_name"] = "Воронеж Магазин (Плехановская, 13)"
    shop_df.at[52, "shop_name"] = "Интернет-магазин Цифровой_склад 1С-Онлайн"
    shop_df.at[54, "shop_name"] = "Якутск Магазин Орджоникидзе, 56"

    shop_df["shop_city"] = shop_df["shop_name"].apply(lambda x: x.split()[0])
    shop_df["shop_type"] = shop_df["shop_name"].apply(lambda x: x.split()[1])

    shop_df = label_encode_feat(shop_df, ["shop_city", "shop_type"])
    hole_df = hole_df.merge(shop_df, how="left", on="shop_id").drop(columns="shop_name")
    return shop_df, hole_df


def add_category_features(
    hole_df: pd.DataFrame, item_category: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_category["sub_category"] = item_category["item_category_name"].apply(
        lambda x: x.split(" - ")[0]
    )
    item_category = label_encode_feat(item_category, ["sub_category"], drop=False)

    hole_df = hole_df.merge(item_category, how="left", on="item_category_id").drop(
        columns="item_category_name"
    )
    hole_df = hole_df.drop(columns="sub_category")
    return item_category, hole_df


def make_pair_feat(hole_data: DataFrame) -> pd.DataFrame:
    df = hole_data.sort_values(["shop_id", "item_id", "date_block_num"])

    # make shop item cnt lag
    # make shop sub_category cnt lag
    # make sub_category item cnt lag
    # make shop-type item cnt lag
    # make shop-city item cnt lag
    new_names = [
        "item_shop_cnt_month_lag",
        "sub-category_shop_cnt_month_lag",
        "sub-category_item_cnt_month_lag",
        "item_shop-type_cnt_month_lag",
        "item_shop-city_cnt_month_lag",
    ]
    df = make_collab_lag(
        df,
        names=[
            ["shop_id", "item_id"],
            ["shop_id", "sub_category_encoded"],
            ["sub_category_encoded", "item_id"],
            ["shop_type_encoded", "item_id"],
            ["shop_city_encoded", "item_id"],
        ],
        new_names=new_names,
        lags=[1, 2, 3],
    )

    df = df.sort_index()

    # make trends for above features
    df = make_trends(df, new_names, [(1, 2)])

    # make ratio for above features
    df = ratio_lag_mean_1_3_to_lag_1(df, new_names)
    return df


def divide_train_test(hole_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    sales_train_df = hole_df[hole_df["ID"] == -1]
    sales_train_df = sales_train_df.drop(columns=["ID", "mean_month_price"])
    test_df = hole_df[hole_df["date_block_num"] == 34]
    test_df = test_df.drop(
        columns=["item_cnt_month", "date_block_num", "mean_month_price"]
    )
    return sales_train_df, test_df


def fix_data(
    sales_df: DataFrame,
    items_df: DataFrame,
    shop_df: DataFrame,
    test_df: DataFrame,
    category_df: DataFrame,
    is_human_in_loop: bool,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Preprocesses dataframes. Copes with outliers. Bring into the correct form.

    :param is_human_in_loop:
    :param sales_df:
    :param items_df:
    :param shop_df:
    :param test_df:
    :param category_df:
    :return: tuple of preprocessed DataFrames
    """

    # Check negative price
    negative_mask = sales_df["item_price"] < 0
    negative_count = negative_mask.sum()
    if negative_count > 0:
        print(f"Negative prices found ({negative_count})")
        sales_df = sales_df[~negative_mask]

    # equal shop names
    sales_df, shop_df, test_df = delete_equal_shop_name(
        sales_df, shop_df, test_df, is_human_in_loop
    )

    # Add category id in data
    sales_df = sales_df.join(items_df["item_category_id"], on="item_id")
    test_df = test_df.join(items_df["item_category_id"], on="item_id")

    # Move to cnt_month
    sales_df = from_day_to_month(sales_df)

    # Remove outliers
    sales_df = remove_outliers(sales_df)

    hole_data = make_single_dataset(sales_df, test_df)

    # add shop_city and shop_type features
    shop_df, hole_data = add_shop_features(hole_data, shop_df)

    # add sub_category feature in main dataset
    category_df, hole_data = add_category_features(hole_data, category_df)

    # add lags for columns
    hole_data = create_lags(
        hole_data,
        [
            "item_id",
            "item_category_id",
            "shop_city_encoded",
            "shop_type_encoded",
            "sub_category_encoded",
            "shop_id",
        ],
        "mean_month_price",  # lag by this column
        [
            "item_price_lag",
            "category_price_lag",
            "shop_city_price_lag",
            "shop_type_price_lag",
            "sub_category_price_lag",
            "shop_price_lag",
        ],
        lag_shifts=[1, 2, 3],
        # insert_NAN_col=False
    )
    hole_data = create_lags(
        hole_data,
        [
            "item_id",
            "item_category_id",
            "shop_city_encoded",
            "shop_type_encoded",
            "sub_category_encoded",
            "shop_id",
        ],
        "item_cnt_month",  # lag by this column
        [
            "item_cnt_lag",
            "category_cnt_lag",
            "shop_city_cnt_lag",
            "shop_type_cnt_lag",
            "sub_category_cnt_lag",
            "shop_cnt_lag",
        ],
        lag_shifts=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        insert_NAN_col=False,
    )

    # make month_since_first_sale
    first_sale_month = (
        hole_data[hole_data["item_price_lag_1_NAN"] == 1]
        .groupby(["item_id"])["date_block_num"]
        .first()
        .rename("months_since_first_sale")
        .reset_index()
    )
    hole_data = hole_data.merge(first_sale_month, how="left", on=["item_id"])
    hole_data["months_since_first_sale"] = (
        hole_data["date_block_num"] - hole_data["months_since_first_sale"]
    )

    # drop NAN columns (it proved useless)
    for column in hole_data.columns:
        if "_NAN" in column:
            hole_data = hole_data.drop(columns=[column])

    # make trend features (lag_1 - lag_2 and lag_1 - lag_12) for above columns
    hole_data = make_trends(
        hole_data,
        [
            "item_cnt_lag",
            "item_price_lag",
            "sub_category_price_lag",
            "sub_category_cnt_lag",
            "category_cnt_lag",
            "shop_city_cnt_lag",
            "shop_type_cnt_lag",
        ],
        [(1, 2)],
    )
    hole_data = make_trends(
        hole_data,
        [
            "item_cnt_lag",
            "sub_category_cnt_lag",
            "category_cnt_lag",
            "shop_city_cnt_lag",
            "shop_type_cnt_lag",
        ],
        [(1, 12)],
    )

    # make feature ratio lag mean to the lag_1
    hole_data = ratio_lag_mean_1_3_to_lag_1(
        hole_data,
        [
            "item_cnt_lag",
            "item_price_lag",
            "sub_category_cnt_lag",
            "category_cnt_lag",
            "shop_city_cnt_lag",
            "shop_type_cnt_lag",
        ],
    )

    # remove reduntant columns lags for below columns from 4 to 11
    delete_lags = [
        "item_cnt_lag",
        "category_cnt_lag",
        "shop_city_cnt_lag",
        "shop_type_cnt_lag",
        "sub_category_cnt_lag",
        "shop_cnt_lag",
    ]
    to_delete = []
    for col in delete_lags:
        for i in range(4, 12):
            to_delete.append(col + "_" + str(i))
    hole_data = hole_data.drop(columns=to_delete)

    # add revenue features
    hole_data["item_revenue_lag_1"] = (
        hole_data["item_cnt_lag_1"] * hole_data["item_price_lag_1"]
    )
    hole_data["item_revenue_lag_2"] = (
        hole_data["item_cnt_lag_2"] * hole_data["item_price_lag_2"]
    )

    hole_data["shop_revenue_lag_1"] = (
        hole_data["shop_cnt_lag_1"] * hole_data["shop_price_lag_1"]
    )
    hole_data["shop_revenue_lag_2"] = (
        hole_data["shop_cnt_lag_2"] * hole_data["shop_price_lag_2"]
    )
    hole_data = make_trends(
        hole_data, ["shop_revenue_lag", "item_revenue_lag"], [(1, 2)]
    )

    # add item_cnt_sales percent of shop sales
    hole_data["item_percent_of_shop_cnt"] = hole_data["item_cnt_lag_1"] / (
        hole_data["shop_cnt_lag_1"] + 1e-4
    )

    hole_data = make_pair_feat(hole_data)

    sales_df, test_df = divide_train_test(hole_data)

    return sales_df, shop_df, test_df, category_df


def ETL(
    data_read_dir: str = "../../data",
    data_write_dir: str = "../../data/preprocessed",
    is_human_in_loop: bool = True,
):
    """
    Read data, fix it and load correct version into memory
    Read:
        items.csv
        sales_train.csv
        shops.csv
        test.csv
        item_categories.csv

    :return: None
    """
    # read data
    items_df = pd.read_csv(data_read_dir + "/items.csv")
    sales_train_df = pd.read_csv(data_read_dir + "/sales_train.csv")
    shops_df = pd.read_csv(data_read_dir + "/shops.csv")
    test_df = pd.read_csv(data_read_dir + "/test.csv")
    category_df = pd.read_csv(data_read_dir + "/item_categories.csv")

    # fix
    sales_train_df_fix, shops_df_fix, test_df_fix, category_df_fix = fix_data(
        sales_train_df, items_df, shops_df, test_df, category_df, is_human_in_loop
    )

    # convert to calculate percent of deleted data
    sales_train_df_old = from_day_to_month(
        sales_train_df.join(items_df["item_category_id"], on="item_id")
    )
    percent = 1 - len(sales_train_df_fix) / len(sales_train_df_old)
    print(f"Was deleted {percent: .2%} of data")

    sales_train_df_fix.to_csv(
        data_write_dir + "/sales_train_preprocessed.csv", index=False
    )
    shops_df_fix.to_csv(data_write_dir + "/shops_preprocessed.csv", index=False)
    test_df_fix.to_csv(data_write_dir + "/test_preprocessed.csv", index=False)
    category_df_fix.to_csv(
        data_write_dir + "/item_categories_preprocessed.csv", index=False
    )
    return


if __name__ == "__main__":
    ETL()
