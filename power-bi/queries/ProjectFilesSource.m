// Staging query name: ProjectFilesSource. Disable load for this query.
// Desktop refresh of this staging design remains pending.
let
    ProjectRoot = "C:\Users\6c10r\Personal\02_projects\01_data\FYP_project",
    Source = Folder.Files(ProjectRoot),
    RequiredFiles = {
        ProjectRoot & "\data\raw\train.csv",
        ProjectRoot & "\outputs\bi_holdout_predictions.csv",
        ProjectRoot & "\outputs\bi_kpis.json",
        ProjectRoot & "\outputs\bi_action_segments.csv"
    },
    Selected = Table.SelectRows(Source, each List.Contains(RequiredFiles, [Folder Path] & [Name]))
in
    Selected
