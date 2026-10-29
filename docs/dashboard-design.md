# Current portfolio dashboard

The primary dashboard for this portfolio is [Customer Churn.portfolio.pbix](<../power-bi/Customer Churn.portfolio.pbix>). Both pages were opened in Power BI Desktop and exported to a [two-page preview](../power-bi/dashboard-design-preview.pdf).

Design accepted for the portfolio on 2026-10-30. The root README uses the new overview preview and includes the second page in an expandable section.

## Changes

- Neutral canvas, white cards, dark navy text and one teal chart color; amber emphasizes observed churn.
- Consistent typography, lighter borders, no heavy shadows, shorter chart titles and aligned margins.
- Overview reduced to four main segment charts. The redundant Medium+ risk card and two model-comparison charts are hidden; their bindings are retained.
- Observed customer metrics and model estimates use separate rows and explicit labels.
- Larger chart areas show all payment and tenure categories. Recommendations are framed as ideas to test.
- The existing sample-size filter on the segment table is retained.

The original PBIX files and screenshot images are preserved. The embedded model, metadata and settings are unchanged; existing measure bindings, filters and navigation remain in the design copy. The [theme JSON](../power-bi/churn-portfolio-theme.json) can be reused, but importing the theme alone does not recreate the layout.

## Data status

These are redesigned views of the historical cached dataset, not refreshed holdout results. The earlier Formula.Firewall issue and staging-query integration remain pending, as requested. Dashboard values therefore differ from the latest Python holdout summary. Follow the [Desktop refresh checklist](power-bi-final-validation.md) when resuming data integration.

A PDF is a static preview; use the PBIX for filtering and navigation. Tables can still scroll to additional rows.
