// Power Query (M) — Load and transform train.csv for churn dashboard
// Paste into Power BI Desktop: Home → Transform data → Advanced Editor
// First create ProjectFilesSource from its companion M file; disable its load.
// Update ProjectRoot here AND in ProjectFilesSource when changing the clone path.
// Staging design: syntax reviewed; successful Desktop refresh is still pending.

let
    // --- Parameter: project root (folder containing data/raw/train.csv) ---
    ProjectRoot = "C:\Users\6c10r\Personal\02_projects\01_data\FYP_project",

    // One folder connector. Downstream steps transform only its binary values.
    // Select exact paths and keep existing privacy settings unchanged.
    ProjectFiles = ProjectFilesSource,
    ReadProjectFile = (relativePath as text) as binary =>
        let
            expectedPath = ProjectRoot & "\" & relativePath,
            matches = Table.SelectRows(ProjectFiles, each [Folder Path] & [Name] = expectedPath)
        in
            if Table.RowCount(matches) <> 1 then error "Expected exactly one project file: " & relativePath
            else matches{0}[Content],

    // --- Load CSV ---
    Source = Csv.Document(
        ReadProjectFile("data\raw\train.csv"),
        [Delimiter = ",", Columns = 21, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),

    // --- Set column types (TotalCharges cleaned in next step) ---
    TypedColumns = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"id", Int64.Type},
            {"gender", type text},
            {"SeniorCitizen", type number},
            {"Partner", type text},
            {"Dependents", type text},
            {"tenure", type number},
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

    // Validate binary SeniorCitizen before integer conversion (no rounding).
    ValidatedSeniorCitizen = Table.TransformColumns(
        TypedColumns,
        {{"SeniorCitizen", each if _ = 0 or _ = 1 then _ else error "SeniorCitizen must be 0 or 1.", Int64.Type}}
    ),
    // Missing TotalCharges is zero only for a zero-month customer; corruption fails.
    WithValidatedCharges = Table.AddColumn(
        ValidatedSeniorCitizen,
        "ValidatedTotalCharges",
        each
            let
                chargeText = if [TotalCharges] = null then "" else Text.Trim(Text.From([TotalCharges])),
                parsed =
                    if chargeText = "" then
                        if [tenure] = 0 then 0 else error "Missing TotalCharges requires tenure=0."
                    else Number.FromText(chargeText, "en-US")
            in
                if [MonthlyCharges] = null or [MonthlyCharges] < 0 then error "MonthlyCharges must be non-negative."
                else if parsed < 0 then error "TotalCharges must be non-negative."
                else parsed,
        type number
    ),
    FilledTotalCharges = Table.RenameColumns(
        Table.RemoveColumns(WithValidatedCharges, {"TotalCharges"}),
        {{"ValidatedTotalCharges", "TotalCharges"}}
    ),

    // --- Tenure bucket for slicers and matrix (matches EDA notebook) ---
    // Bins: 0-6, 6-12, 12-24, 24-60, 60+ months
    AddedTenureBucket = Table.AddColumn(
        FilledTotalCharges,
        "tenure_bucket",
        each
            if [tenure] = null or [tenure] < 0 or [tenure] > 100 or Number.Mod([tenure], 1) <> 0 then
                error "tenure must contain whole months between 0 and 100, with no missing values."
            else if [tenure] <= 6 then "0-6"
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

    // Load the same run's policy; this is not a hard-coded visual-only threshold.
    EvaluationMetadata = Json.Document(ReadProjectFile("outputs\bi_kpis.json")),

    // --- Join ML predictions (run: python -m src.export_predictions) ---
    PredictionsSource = Csv.Document(
        ReadProjectFile("outputs\bi_holdout_predictions.csv"),
        [Delimiter = ",", Columns = 7, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    PredictionsHeaders = Table.PromoteHeaders(PredictionsSource, [PromoteAllScalars = true]),
    TypedPredictions = Table.TransformColumnTypes(
        PredictionsHeaders,
        {
            {"id", Int64.Type},
            {"Churn", type text},
            {"predicted_proba", type number},
            {"risk_band", type text},
            {"prediction_scope", type text},
            {"model_version", type text},
            {"run_id", type text}
        }
    ),
    // Historical ML evaluation is explicitly restricted to the held-out population.
    ValidatedPredictions =
        if Table.RowCount(TypedPredictions) <> Table.RowCount(Table.Distinct(TypedPredictions, {"id"})) then
            error "Duplicate prediction IDs."
        else if not List.AllTrue(List.Transform(TypedPredictions[prediction_scope], each _ = "holdout")) then
            error "BI historical evaluation requires holdout predictions."
        else if List.Count(List.Distinct(TypedPredictions[run_id])) <> 1 or List.Count(List.Distinct(TypedPredictions[model_version])) <> 1 then
            error "Predictions must belong to one run and one evaluation model."
        else if List.First(TypedPredictions[run_id]) <> EvaluationMetadata[run_id] then
            error "Prediction and KPI manifests belong to different runs."
        else TypedPredictions,
    PredictionsSlim = Table.SelectColumns(ValidatedPredictions, {"id", "predicted_proba", "risk_band", "prediction_scope", "model_version", "run_id"}),

    MergedPredictions = Table.NestedJoin(
        AddedSeniorCitizenLabel,
        {"id"},
        PredictionsSlim,
        {"id"},
        "ml_pred",
        JoinKind.Inner
    ),
    CheckedJoin =
        if Table.RowCount(MergedPredictions) <> Table.RowCount(PredictionsSlim) then
            error "Holdout prediction IDs do not match unique source rows."
        else MergedPredictions,
    ExpandedPredictions = Table.ExpandTableColumn(
        CheckedJoin,
        "ml_pred",
        {"predicted_proba", "risk_band", "prediction_scope", "model_version", "run_id"},
        {"predicted_proba", "risk_band", "prediction_scope", "model_version", "run_id"}
    ),

    WithSamplePolicy = Table.AddColumn(
        ExpandedPredictions,
        "segment_min_count",
        each EvaluationMetadata[min_segment_count],
        Int64.Type
    ),

    // --- Risk band sort order for slicers (Low → Medium → High) ---
    AddedRiskBandSort = Table.AddColumn(
        WithSamplePolicy,
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
