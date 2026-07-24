# Data Understanding Report

## Dataset Shape
- **Rows:** 4,009
- **Columns:** 12

## Columns

- `brand` (str)
- `model` (str)
- `model_year` (int64)
- `milage` (str)
- `fuel_type` (str)
- `engine` (str)
- `transmission` (str)
- `ext_col` (str)
- `int_col` (str)
- `accident` (str)
- `clean_title` (str)
- `price` (str)

## Missing Values

- **brand:** 0 (0.0%)
- **model:** 0 (0.0%)
- **model_year:** 0 (0.0%)
- **milage:** 0 (0.0%)
- **fuel_type:** 170 (4.24%)
- **engine:** 0 (0.0%)
- **transmission:** 0 (0.0%)
- **ext_col:** 0 (0.0%)
- **int_col:** 0 (0.0%)
- **accident:** 113 (2.82%)
- **clean_title:** 596 (14.87%)
- **price:** 0 (0.0%)

## Duplicated Rows
- **Count:** 0

## Descriptive Statistics

```
       brand    model   model_year       milage fuel_type                      engine transmission ext_col int_col       accident clean_title    price
count   4009     4009  4009.000000         4009      3839                        4009         4009    4009    4009           3896        3413     4009
unique    57     1898          NaN         2818         7                        1146           62     319     156              2           1     1569
top     Ford  M3 Base          NaN  110,000 mi.  Gasoline  2.0L I4 16V GDI DOHC Turbo          A/T   Black   Black  None reported         Yes  $15,000
freq     386       30          NaN           16      3309                          52         1037     905    2025           2910        3413       39
mean     NaN      NaN  2015.515590          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
std      NaN      NaN     6.104816          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
min      NaN      NaN  1974.000000          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
25%      NaN      NaN  2012.000000          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
50%      NaN      NaN  2017.000000          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
75%      NaN      NaN  2020.000000          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
max      NaN      NaN  2024.000000          NaN       NaN                         NaN          NaN     NaN     NaN            NaN         NaN      NaN
```
