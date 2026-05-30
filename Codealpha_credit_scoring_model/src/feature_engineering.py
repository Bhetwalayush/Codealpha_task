def add_features(df):

    df["debt_to_income_ratio"] = (
        df["debt"] / df["income"]
    )

    df["loan_to_income_ratio"] = (
        df["loan_amount"] / df["income"]
    )

    return df