# ---
#
#
# ---

# %%
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_theme(style='white')
sns.set_palette('cividis')

plt.rcParams['figure.figsize'] = [12, 8]

plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.1

plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.color'] = '#ff00ff'
plt.rcParams['grid.linewidth'] = 0.5

plt.rcParams['legend.framealpha'] = 0.5
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['legend.loc'] = 'best'  # noqa: F821
plt.rcParams['legend.fontsize'] = 'small'  # noqa: F821

# %%
parent_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(parent_dir, 'data', 'raw')
figure_path = os.path.join(parent_dir, 'figures')

train = pl.read_csv(source=os.path.join(data_path, 'train.csv'), dtypes={'StateHoliday': pl.String, 'Date': pl.Date})
store = pl.read_csv(source=os.path.join(data_path, 'store.csv'), ignore_errors=True)
test = pl.read_csv(source=os.path.join(data_path, 'test.csv'), ignore_errors=True)

train = train.with_columns(
    pl.col('Date').dt.year().cast(pl.Int64).alias('Year'),
    pl.col('Date').dt.month().cast(pl.Int64).alias('Month'),
    pl.col('Date').dt.ordinal_day().cast(pl.Int64).alias('DayOfYear'),
    pl.col('Date').dt.day().alias('DayOfMonth'),
)

train = train.with_columns(
    pl.col('Promo').cast(pl.String).cast(pl.Categorical),
    pl.col('StateHoliday').cast(pl.String).cast(pl.Categorical),
    pl.col('SchoolHoliday').cast(pl.String).cast(pl.Categorical),
)

store = store.with_columns(
    pl.col('StoreType').cast(pl.String).cast(pl.Categorical), pl.col('Assortment').cast(pl.String).cast(pl.Categorical)
)

AVG_DAILY_SALES = train.filter(pl.col('Sales') > 0).select('Sales').to_series().mean()
MIN_DAILY_SALES = train.filter(pl.col('Sales') > 0).select('Sales').to_series().min()
MAX_DAILY_SALES = train.filter(pl.col('Sales') > 0).select('Sales').to_series().max()

# %%
print(train.shape)
print(test.shape)
print(store.shape)

# %%
print(train.select('Date').to_series().min())
print(train.select('Date').to_series().max())

print(test.select('Date').to_series().min())
print(test.select('Date').to_series().max())

# %%
print(train.group_by('StateHoliday').len())
print(train.group_by('SchoolHoliday').len())

# %%
print(train.group_by('Open').len())
print(test.group_by('Open').len())

# %%
print(train.group_by('Year').len())
print(train.group_by('Month').len())

# %%
print(train.filter(pl.col('Open') > 0).group_by('Promo').len())
print(train.filter(pl.col('Open') > 0).group_by('StateHoliday').len())
print(train.filter(pl.col('Open') > 0).group_by('SchoolHoliday').len())

# %%
print(
    store.pivot(
        values='StoreType',
        index='Assortment',
        columns='StoreType',
        aggregate_function='len',
    ).fill_null(0)
)

# %% Missing values
plt_data = train.pivot(
    values='Sales',
    index='Date',
    columns='Store',
    aggregate_function='sum',
).with_columns(pl.all().is_null())

sns.heatmap(plt_data)
plt.savefig(os.path.join(figure_path, 'missing_values_train.png'))

# %% Log-normalisation for sales values
plt_data = train.with_columns(pl.col('Sales').truediv(pl.col('Sales').median()).log())
sns.histplot(plt_data, x='Sales')
plt.savefig(os.path.join(figure_path, 'log_normalised_sales_values.png'))

# %% Log-normalisation for competition distance
plt_data = store.with_columns(pl.col('CompetitionDistance').truediv(pl.col('CompetitionDistance').median()).log())
sns.histplot(plt_data, x='CompetitionDistance', bins=100)
plt.savefig(os.path.join(figure_path, 'log_normalised_competition_dist.png'))

# %% Relationship of customers vs Sales
plt_data = train.sample(n=100_000)
sns.scatterplot(plt_data, x='Customers', y='Sales', s=1)
plt.savefig(os.path.join(figure_path, 'customer_vs_sales_scatter.png'))

# %% Linear elasticities (given log-scale)
plt_data = train.sample(n=100_000).select(['Customers', 'Sales']).with_columns(pl.all().log())
sns.scatterplot(plt_data, x='Customers', y='Sales', s=1)

# %%
print(train.filter(pl.col('Sales') == 0).select('Customers').to_series().mean())

# %%
print(train.group_by('Promo').agg(pl.col('Sales').mean()))

# %%
print(train.group_by('StateHoliday').agg(pl.col('Sales').mean()))

# %%
print(train.group_by('SchoolHoliday').agg(pl.col('Sales').mean()))

# %% Plot individual time series only for the open days
SAMPLE_STORES = [174, 346, 693, 934]

plt_data = train.filter(pl.col('Store').is_in(SAMPLE_STORES)).filter(pl.col('Open').gt(0))

g = sns.FacetGrid(plt_data, col='Store', col_wrap=1, height=4, aspect=3)
g.map_dataframe(sns.lineplot, x='Date', y='Sales', linewidth=1)
g.set_axis_labels('Date', 'Sales')
g.set_titles('Store {col_name}')
plt.tight_layout()

plt.savefig(os.path.join(figure_path, 'individual_stores_time_series.png'))
plt.show()

# %%
SAMPLE_STORES = [174, 346, 693, 934]

plt_data = (
    train.filter(pl.col('Store').is_in(SAMPLE_STORES))
    .filter(pl.col('Open').gt(0))
    .with_columns(pl.col('Sales').truediv(AVG_DAILY_SALES))
)

g = sns.FacetGrid(plt_data, col='Store', col_wrap=1, height=4, aspect=3)
g.map_dataframe(sns.lineplot, x='Date', y='Sales', linewidth=1)
g.set_axis_labels('Date', 'Sales')
g.set_titles('Store {col_name}')
plt.tight_layout()

plt.savefig(os.path.join(figure_path, 'individual_stores_time_series_norm.png'))
plt.show()

# %% Check the daily or monthly effect on the daily sales
plt_data = (
    train.filter(pl.col('Open') > 0)
    .group_by(['DayOfYear'])
    .agg(pl.col('Sales').mean().alias('avg_sales'))
    .with_columns(pl.col('avg_sales').truediv(AVG_DAILY_SALES).alias('avg_sales_rel'))
)
sns.lineplot(plt_data, x='DayOfYear', y='avg_sales_rel', linewidth=1)
plt.savefig(os.path.join(figure_path, 'doy_effect_on_avg_sales.png'))

# %% same, but by store
plt_data = (
    train.filter(pl.col('Open') > 0)
    .with_columns(pl.col('Sales').mean().over('Store').alias('avg_sales_by_store'))
    .group_by(['Store', 'DayOfYear', 'avg_sales_by_store'])
    .agg(pl.col('Sales').mean().alias('avg_sales'))
    .with_columns(pl.col('avg_sales').truediv(AVG_DAILY_SALES).alias('avg_sales_rel'))
)
sns.scatterplot(plt_data, x='DayOfYear', y='avg_sales_rel', s=1)
plt.savefig(os.path.join(figure_path, 'doy_effect_on_avg_sales_bystore.png'))

# %% Day of month effect seems very strong, with start/end + mid-of-month higher than avg sales
plt_data = (
    train.filter(pl.col('Open') > 0)
    .with_columns(pl.col('Sales').mean().over('Store').alias('avg_sales_by_store'))
    .group_by(['Store', 'DayOfMonth', 'avg_sales_by_store'])
    .agg(pl.col('Sales').mean().alias('avg_sales'))
    .with_columns(pl.col('avg_sales').truediv(pl.col('avg_sales_by_store')).alias('avg_sales_rel'))
)
sns.scatterplot(plt_data, x='DayOfMonth', y='avg_sales_rel', s=1)
plt.savefig(os.path.join(figure_path, 'dom_effect_on_avg_sales_bystore.png'))

# %%
plt_data = (
    train.join(store, on='Store', how='left')
    .group_by('Store', 'Month', 'StoreType')
    .agg(pl.col('Sales').mean().alias('avg_monthly_sales'))
)
sns.histplot(plt_data, x='avg_monthly_sales', hue='StoreType', bins=100)
plt.savefig(os.path.join(figure_path, 'avg_monthly_sales_by_storetype.png'))

# %%
plt_data = (
    train.join(store, on='Store', how='left')
    .group_by('Store', 'Month', 'Assortment')
    .agg(pl.col('Sales').mean().alias('avg_monthly_sales'))
)
sns.histplot(plt_data, x='avg_monthly_sales', hue='Assortment', bins=100)
plt.savefig(os.path.join(figure_path, 'avg_monthly_sales_by_assortment.png'))

# %%
plt_data = (
    train.join(store, on='Store', how='left')
    .group_by('Store', 'Month', 'Promo')
    .agg(pl.col('Sales').mean().alias('avg_monthly_sales'))
)
sns.histplot(plt_data, x='avg_monthly_sales', hue='Promo', bins=100)
plt.savefig(os.path.join(figure_path, 'avg_monthly_sales_by_promo.png'))

# %%
plt_data = (
    train.join(store, on='Store', how='left')
    .group_by('Store', 'Month', 'Promo2')
    .agg(pl.col('Sales').mean().alias('avg_monthly_sales'))
)
sns.histplot(plt_data, x='avg_monthly_sales', hue='Promo2', bins=100)
plt.savefig(os.path.join(figure_path, 'avg_monthly_sales_by_promo2.png'))

# %%
