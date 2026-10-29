// Optional separate action table: only sample-qualified holdout segments are exported.
let
    ProjectRoot = "C:\Users\6c10r\Personal\02_projects\01_data\FYP_project",
    // Only reference the staging query; do not mix a direct source with references.
    ReadProjectFile = (relativePath as text) as binary =>
        let
            expectedPath = ProjectRoot & "\" & relativePath,
            matches = Table.SelectRows(ProjectFilesSource, each [Folder Path] & [Name] = expectedPath)
        in
            if Table.RowCount(matches) <> 1 then error "Expected exactly one project file: " & relativePath
            else matches{0}[Content],
    Metadata = Json.Document(ReadProjectFile("outputs\bi_kpis.json")),
    Source = Csv.Document(ReadProjectFile("outputs\bi_action_segments.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Headers, {{"action_rank", Int64.Type}, {"customer_count", Int64.Type}, {"churned_count", Int64.Type}, {"observed_churn_rate", type number}, {"mean_predicted_score", type number}, {"predicted_high_risk_count", Int64.Type}}),
    Checked =
        if not List.AllTrue(List.Transform(Typed[customer_count], each _ >= Metadata[min_segment_count])) then
            error "Action table contains an undersized segment."
        else if not List.AllTrue(List.Transform(Typed[prediction_scope], each _ = "holdout")) then
            error "Action table must use holdout scores."
        else if not List.AllTrue(List.Transform(Typed[run_id], each _ = Metadata[run_id])) then
            error "Action segments and KPI metadata belong to different runs."
        else if not List.AllTrue(List.Transform(Typed[model_version], each _ = Metadata[model_version])) then
            error "Action segments use a different evaluation model."
        else Typed
in
    Checked
