# Ames Housing — Feature-by-Feature Guide

Based on `train.csv` (1,460 houses, 81 columns). For each feature: what it means,
in plain English, plus the actual values/range found in this dataset.

> **Quality/condition code legend** (used by many columns below):
> `Ex` = Excellent, `Gd` = Good, `TA` = Average/Typical, `Fa` = Fair, `Po` = Poor

---

## Identification & Zoning

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **Id** | Row number, not a real feature | 1 to 1460 |
| **MSSubClass** | Code for the building's class (age/style combo, e.g. 20 = 1-story 1946+) | 20 to 190 |
| **MSZoning** | General zoning: residential, commercial, etc. | RL (Residential Low), RM, C (all) [Commercial], FV (Floating Village), RH |

## Lot Characteristics

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **LotFrontage** | Feet of street frontage | 21 – 313 ft (259 houses have this missing) |
| **LotArea** | Lot size | 1,300 – 215,245 sq ft |
| **Street** | Road type to the house | Pave, Grvl |
| **Alley** | Alley access type | Grvl, Pave (1,369 houses have no alley) |
| **LotShape** | Regularity of the lot's shape | Reg (regular), IR1/IR2/IR3 (increasingly irregular) |
| **LandContour** | Flatness of the property | Lvl (level), Bnk (banked), Low, HLS (hillside) |
| **Utilities** | Utilities available | AllPub (all public), NoSeWa (no sewer/water) |
| **LotConfig** | Lot position | Inside, Corner, CulDSac, FR2/FR3 (fronts 2/3 sides) |
| **LandSlope** | Steepness of the land | Gtl (gentle), Mod (moderate), Sev (severe) |

## Location

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **Neighborhood** | Which part of Ames | 25 neighborhoods, e.g. CollgCr, OldTown, NridgHt, Edwards |
| **Condition1** | Nearby feature affecting the property | Norm, Feedr (feeder street), Artery, RR (railroad), Pos (positive, e.g. near park) |
| **Condition2** | A second nearby feature, if any | Same categories as Condition1 |

## Building Type & Style

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **BldgType** | Kind of dwelling | 1Fam (single-family), 2fmCon, Duplex, TwnhsE, Twnhs |
| **HouseStyle** | Floor layout | 1Story, 2Story, 1.5Fin/Unf, SFoyer, SLvl, 2.5Fin/Unf |

## Overall Ratings & Age

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **OverallQual** | Overall material/finish quality, 1 (worst) – 10 (best) | 1 – 10 |
| **OverallCond** | Overall condition, 1 (worst) – 9 (best, in this dataset) | 1 – 9 |
| **YearBuilt** | Year originally built | 1872 – 2010 |
| **YearRemodAdd** | Year of last remodel (= YearBuilt if never remodeled) | 1950 – 2010 |

## Roof & Exterior

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **RoofStyle** | Roof shape | Gable, Hip, Gambrel, Mansard, Flat, Shed |
| **RoofMatl** | Roof material | CompShg (composite shingle), WdShngl, Metal, WdShake, Membran, Tar&Grv, Roll, ClyTile |
| **Exterior1st / Exterior2nd** | Outer wall material(s) — 2nd if more than one is used | VinylSd, MetalSd, Wd Sdng, HdBoard, BrkFace, Plywood, Stucco, etc. |
| **MasVnrType** | Masonry veneer type (decorative stone/brick layer) | BrkFace, Stone, BrkCmn (872 houses have none) |
| **MasVnrArea** | Size of that veneer | 0 – 1,600 sq ft (8 missing) |
| **ExterQual** | Quality of exterior material | Ex, Gd, TA, Fa |
| **ExterCond** | Present condition of exterior material | Ex, Gd, TA, Fa, Po |
| **Foundation** | What the house sits on | PConc (poured concrete), CBlock, BrkTil, Wood, Slab, Stone |

## Basement

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **BsmtQual** | Basement height/quality | Ex, Gd, TA, Fa (37 houses have no basement) |
| **BsmtCond** | General basement condition | TA, Gd, Fa, Po |
| **BsmtExposure** | Walkout/garden-level exposure | No, Mn (minimal), Av (average), Gd |
| **BsmtFinType1** | Quality of main finished basement area | GLQ (good living quarters) down to Unf (unfinished) |
| **BsmtFinSF1** | Size of that finished area | 0 – 5,644 sq ft |
| **BsmtFinType2** | Quality of a second finished basement area, if any | Same scale as Type1 |
| **BsmtFinSF2** | Size of that second area | 0 – 1,474 sq ft |
| **BsmtUnfSF** | Unfinished basement square footage | 0 – 2,336 sq ft |
| **TotalBsmtSF** | Total basement size | 0 – 6,110 sq ft |

## Heating & Utilities

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **Heating** | Heating system type | GasA (gas forced air), GasW, Grav, Wall, OthW, Floor |
| **HeatingQC** | Heating quality/condition | Ex, Gd, TA, Fa, Po |
| **CentralAir** | Central A/C present? | Y, N |
| **Electrical** | Electrical system type | SBrkr (breakers), FuseF/A/P (fuses), Mix |

## Living Space

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **1stFlrSF** | First floor square footage | 334 – 4,692 |
| **2ndFlrSF** | Second floor square footage | 0 – 2,065 |
| **LowQualFinSF** | Low-quality finished square footage | 0 – 572 |
| **GrLivArea** | Total above-ground living area — usually the single biggest price driver | 334 – 5,642 |

## Bathrooms & Rooms

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **BsmtFullBath / BsmtHalfBath** | Full/half bathrooms in the basement | 0–3 full, 0–2 half |
| **FullBath / HalfBath** | Full/half bathrooms above ground | 0–3 full, 0–2 half |
| **BedroomAbvGr** | Bedrooms above ground | 0 – 8 |
| **KitchenAbvGr** | Kitchens above ground | 0 – 3 (almost always 1) |
| **KitchenQual** | Kitchen quality | Ex, Gd, TA, Fa |
| **TotRmsAbvGrd** | Total rooms above ground (excludes bathrooms) | 2 – 14 |
| **Functional** | Deductions for functional issues | Typ (typical/no issues) down to Sev (severe) |

## Fireplace

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **Fireplaces** | Number of fireplaces | 0 – 3 |
| **FireplaceQu** | Fireplace quality | Ex, Gd, TA, Fa, Po (690 houses have none) |

## Garage

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **GarageType** | Garage location/type | Attchd, Detchd, BuiltIn, CarPort, Basment, 2Types (81 have none) |
| **GarageYrBlt** | Year garage was built | 1900 – 2010 |
| **GarageFinish** | Interior finish level | RFn (rough finished), Unf, Fin |
| **GarageCars** | Car capacity | 0 – 4 |
| **GarageArea** | Garage size | 0 – 1,418 sq ft |
| **GarageQual** | Garage quality | TA, Fa, Gd, Ex, Po |
| **GarageCond** | Garage condition | TA, Fa, Gd, Po, Ex |
| **PavedDrive** | Driveway surface | Y (paved), N (dirt/gravel), P (partial) |

## Outdoor Features

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **WoodDeckSF** | Wood deck size | 0 – 857 sq ft |
| **OpenPorchSF** | Open porch size | 0 – 547 sq ft |
| **EnclosedPorch** | Enclosed porch size | 0 – 552 sq ft |
| **3SsnPorch** | Three-season porch size | 0 – 508 sq ft |
| **ScreenPorch** | Screened porch size | 0 – 480 sq ft |
| **PoolArea** | Pool size | 0 – 738 sq ft (almost all houses have none) |
| **PoolQC** | Pool quality | Ex, Gd, Fa (1,453 of 1,460 houses have no pool) |
| **Fence** | Fence quality | MnPrv, GdWo, GdPrv, MnWw (1,179 have no fence) |
| **MiscFeature** | Unusual extra (shed, 2nd garage, tennis court, etc.) | Shed, Gar2, Othr, TenC (1,406 have none) |
| **MiscVal** | Dollar value of that extra feature | 0 – 15,500 |

## Sale Details

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **MoSold** | Month sold | 1 – 12 |
| **YrSold** | Year sold | 2006 – 2010 |
| **SaleType** | How the sale happened | WD (warranty deed/normal), New, COD, Con family variants |
| **SaleCondition** | Circumstances of sale | Normal, Abnorml (foreclosure/short sale), Partial (new build), Family, AdjLand, Alloca |

## Target Variable

| Feature | Meaning | What's in this dataset |
|---|---|---|
| **SalePrice** | The actual sale price — what your model predicts | $34,900 – $755,000 |

---

## Missing-value quick reference

A few features have a lot of missing values — but in most cases "missing" simply
means the house **doesn't have that feature** (e.g. no pool, no alley), not that
data was lost:

| Feature | Missing count | Likely reason |
|---|---|---|
| PoolQC | 1,453 | No pool |
| MiscFeature | 1,406 | No special extra |
| Alley | 1,369 | No alley access |
| Fence | 1,179 | No fence |
| FireplaceQu | 690 | No fireplace |
| LotFrontage | 259 | Not recorded / irregular lot |
| GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond | 81 | No garage |
| BsmtExposure, BsmtFinType2 | 38 | No basement |
| BsmtQual, BsmtCond, BsmtFinType1 | 37 | No basement |
| MasVnrType, MasVnrArea | 8 | No masonry veneer |
| Electrical | 1 | Single unrecorded case |
