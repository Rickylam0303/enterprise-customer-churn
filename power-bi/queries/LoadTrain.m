// Power Query (M) — Load and transform train.csv for churn dashboard
// Paste into Power BI Desktop: Home → Transform data → Advanced Editor
// Update ProjectRoot parameter to your local clone path.

let
    // --- Parameter: project root (folder containing data/raw/train.csv) ---
    ProjectRoot = "C:\Users\6c10r\OneDrive\Personal\01_career\03_Portfolio_&_Projects\FYP project",

    // --- Load CSV ---
    Source = Csv.Document(
        File.Contents(ProjectRoot & "\data\raw\train.csv"),
        [Delimiter = ",", Columns = 21, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),

    // --- Set column types (TotalCharges cleaned in next step) ---
    TypedColumns = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"id", Int64.Type},
            {"gender", type text},
            {"SeniorCitizen", Int64.Type},
            {"Partner", type text},
            {"Dependents", type text},
            {"tenure", Int64.Type},
            {"PhoneService", type text},
            {"MultipleLines", type text},
            {"InternetService", type text},
            {"OnlineSecurity", type text},
            {"OnlineBackup", type text},
            {"DeviceProtection", type text},
            {"TechSupport", type text},
            {"StreamingTV", type text},
            {"StreamingMovies", type text},
            {"Contract", type text},
            {"PaperlessBilling", type text},
            {"PaymentMethod", type text},
            {"MonthlyCharges", type number},
            {"TotalCharges", type text},
            {"Churn", type text}
        }
    ),

    // --- Clean TotalCharges (matches src/data.py clean_total_charges) ---
    // Trim whitespace, coerce to number, replace null/blank with 0
    TrimmedTotalCharges = Table.TransformColumns(
        TypedColumns,
        {{"TotalCharges", each Text.Trim(Text.From(_)), type text}}
    ),
    ParsedTotalCharges = Table.TransformColumns(
        TrimmedTotalCharges,
        {{"TotalCharges", each try Number.From(_) otherwise null, type number}}
    ),
    FilledTotalCharges = Table.ReplaceValue(
        ParsedTotalCharges,
        null,
        0,
        Replacer.ReplaceValue,
        {"TotalCharges"}
    ),

    // --- Tenure bucket for slicers and matrix (matches EDA notebook) ---
    // Bins: 0-6, 6-12, 12-24, 24-60, 60+ months
    AddedTenureBucket = Table.AddColumn(
        FilledTotalCharges,
        "tenure_bucket",
        each
            if [tenure] <= 6 then "0-6"
            else if [tenure] <= 12 then "6-12"
            else if [tenure] <= 24 then "12-24"
            else if [tenure] <= 60 then "24-60"
            else "60+",
        type text
    ),

    // --- Sort order for tenure_bucket (use as "Sort by column" in model) ---
    AddedTenureBucketSort = Table.AddColumn(
        AddedTenureBucket,
        "tenure_bucket_sort",
        each
            if [tenure_bucket] = "0-6" then 1
            else if [tenure_bucket] = "6-12" then 2
            else if [tenure_bucket] = "12-24" then 3
            else if [tenure_bucket] = "24-60" then 4
            else 5,
        Int64.Type
    ),

    // --- Service count for Page 2 bundle view (matches src/features.py) ---
    ServiceColumns = {
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    },
    AddedServiceCount = Table.AddColumn(
        AddedTenureBucketSort,
        "service_count",
        each List.Sum(List.Transform(Record.ToList(Record.SelectFields(_, ServiceColumns)), each if _ = "Yes" then 1 else 0)),
        Int64.Type
    ),

    // --- Monthly charge bins for Page 2 price sensitivity chart ---
    AddedMonthlyChargesBucket = Table.AddColumn(
        AddedServiceCount,
        "MonthlyCharges_Bucket",
        each
            if [MonthlyCharges] < 35 then "$0-35"
            else if [MonthlyCharges] < 55 then "$35-55"
            else if [MonthlyCharges] < 75 then "$55-75"
            else if [MonthlyCharges] < 95 then "$75-95"
            else "$95+",
        type text
    ),
    AddedMonthlyChargesBucketSort = Table.AddColumn(
        AddedMonthlyChargesBucket,
        "MonthlyCharges_Bucket_sort",
        each
            if [MonthlyCharges_Bucket] = "$0-35" then 1
            else if [MonthlyCharges_Bucket] = "$35-55" then 2
            else if [MonthlyCharges_Bucket] = "$55-75" then 3
            else if [MonthlyCharges_Bucket] = "$75-95" then 4
            else 5,
        Int64.Type
    ),

    // --- Binary churn flag for optional use in visuals ---
    AddedChurnBinary = Table.AddColumn(
        AddedMonthlyChargesBucketSort,
        "Churn_binary",
        each if [Churn] = "Yes" then 1 else 0,
        Int64.Type
    ),

    // --- Friendly labels ---
    AddedSeniorCitizenLabel = Table.AddColumn(
        AddedChurnBinary,
        "SeniorCitizen_Label",
        each if [SeniorCitizen] = 1 then "Yes" else "No",
        type text
    ),

    // --- Join ML predictions (run: python -m src.export_predictions) ---
    PredictionsSource = Csv.Document(
        File.Contents(ProjectRoot & "\outputs\bi_train_predictions.csv"),
        [Delimiter = ",", Columns = 4, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    PredictionsHeaders = Table.PromoteHeaders(PredictionsSource, [PromoteAllScalars = true]),
    TypedPredictions = Table.TransformColumnTypes(
        PredictionsHeaders,
        {
            {"id", Int64.Type},
            {"Churn", type text},
            {"predicted_proba", type number},
            {"risk_band", type text}
        }
    ),
    PredictionsSlim = Table.SelectColumns(TypedPredictions, {"id", "predicted_proba", "risk_band"}),

    MergedPredictions = Table.NestedJoin(
        AddedSeniorCitizenLabel,
        {"id"},
        PredictionsSlim,
        {"id"},
        "ml_pred",
        JoinKind.LeftOuter
    ),
    ExpandedPredictions = Table.ExpandTableColumn(
        MergedPredictions,
        "ml_pred",
        {"predicted_proba", "risk_band"},
        {"predicted_proba", "risk_band"}
    ),

    // --- Risk band sort order for slicers (Low → Medium → High) ---
    AddedRiskBandSort = Table.AddColumn(
        ExpandedPredictions,
        "risk_band_sort",
        each
            if [risk_band] = "Low" then 1
            else if [risk_band] = "Medium" then 2
            else if [risk_band] = "High" then 3
            else 0,
        Int64.Type
    )
in
    AddedRiskBandSort
