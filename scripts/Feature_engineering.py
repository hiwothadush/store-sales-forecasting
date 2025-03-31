# Extract datetime features
def Extract_datetime_features(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Weekday'] = data['Date'].dt.weekday
    data['IsWeekend'] = data['Weekday'].isin([5, 6]).astype(int)
    data['DaysToNextHoliday'] = data['StateHoliday'].shift(-1).notna().cumsum() - data['StateHoliday'].notna().cumsum()
    data['DaysSinceLastHoliday'] = data['StateHoliday'].notna().cumsum() - data['StateHoliday'].shift(1).notna().cumsum()
    data['MonthPart'] = pd.cut(data['Date'].dt.day, bins=[0, 10, 20, 31], labels=['Start', 'Mid', 'End'])


def one_hot_encode(df, columns):
    """
    Performs one-hot encoding on specified categorical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: A DataFrame with specified categorical variables one-hot encoded.
    """
    # Ensure the provided columns exist in the DataFrame
    columns = [col for col in columns if col in df.columns]

    # Apply one-hot encoding without dropping any categories
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=False)

    return df_encoded