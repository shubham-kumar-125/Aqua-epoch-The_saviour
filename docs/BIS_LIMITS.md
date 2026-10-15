# Water-metric tooltip guidance

The simplified form displays the following **BIS IS 10500:2012** guidance. These values help a user understand individual inputs; they are not a replacement for comprehensive water testing and are not the machine-learning model's only decision rule.

| Metric | Unit | Acceptable limit | Permissible limit when no alternate source is available |
| --- | --- | ---: | ---: |
| Turbidity | NTU | 1 | 5 |
| Total hardness | mg/L as CaCO₃ | 200 | 600 |
| Total dissolved solids (TDS) | mg/L | 500 | 2,000 |

Sources: [BIS IS 10500:2012, Drinking Water — Specification](https://www.bis.gov.in/other/DrinWatIS10500.pdf) and the Government of India’s [Uniform Drinking Water Quality Monitoring Protocol](https://jaljeevanmission.gov.in/sites/default/files/guideline/UniformDrinkingWaterQualityMonitoringProtocol.pdf).

The application validates broad physical form ranges before sending a request: Turbidity 0–1,000 NTU, hardness 0–2,000 mg/L as CaCO₃, and TDS 0–50,000 mg/L. Readings outside a BIS acceptable/permissible limit are shown in the tooltips; they do **not** by themselves certify a sample as unsafe or safe. Confirm any result through accredited laboratory testing.
