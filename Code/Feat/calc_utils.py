
def divide(molecule, denominator):
	if denominator == 0.0:
		return molecule
	return molecule * 1.0 / denominator

def divide_df(df, molecule_column, denominator_column):
	return df.apply(lambda row: divide(row[molecule_column], row[denominator_column]), axis=1)