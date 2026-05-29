import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const baseDir = "C:\\Users\\mosta\\OneDrive\\Desktop\\GraphCast";
const outDir = path.join(baseDir, "nsf_ags_prf_uploads", "personnel_latex");
const templatePath = path.join(outDir, "coa_template_official.xlsx");
const outputPath = path.join(outDir, "collaborators_other_affiliations_mostafa_rezaali.xlsx");

const input = await FileBlob.load(templatePath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("NSF COA Template");

const blank = (rows, cols = 8) => Array.from({ length: rows }, () => Array(cols).fill(null));
sheet.getRange("A17:H95").values = blank(79);

const rows = [
  [null, "Rezaali, Mostafa", "University of Florida", null, null, null, null, null],
  [null, null, "University of Florida, Department of Geography (Ph.D. candidate)", null, null, null, null, null],
  [null, null, null, null, null, null, null, null],

  ["Table 2: List names as last name, first name, middle initial, for whom a personal, family, or business relationship would otherwise preclude their service as a reviewer.", null, null, null, null, null, null, null],
  [null, "Select \"R:\" for column A to designate relationships that would otherwise preclude their service as a reviewer.", null, null, null, null, null, null],
  [null, null, null, "To disambiguate common names", null, null, null, null],
  ["2", "Name", "Type of Relationship", "Optional  (email, Department)", "Last Active", null, null, null],
  [null, null, null, null, null, null, null, null],

  ["Table 3: List names as last name, first name, middle initial, and provide organizational affiliations, if known, for the following:\nG: The individual's Ph.D. advisors; and\nT: All of the individual's Ph.D. thesis advisees.", null, null, null, null, null, null, null],
  [null, null, null, "To disambiguate common names", null, null, null, null],
  ["3", "Advisor/Advisee Name:", "Organizational Affiliation", "Optional  (email, Department)", null, null, null, null],
  ["G:", "Keellings, David", "University of Florida", "Department of Geography", null, null, null, null],
  ["G:", "Karimi, Ali", "Qom University of Technology", null, null, null, null, null],
  [null, null, null, null, null, null, null, null],

  ["Table 4: List names as last name, first name, middle initial, and provide organizational affiliations, if known, for the following:", null, null, null, null, null, null, null],
  ["A: Co-authors on any book, article, report, abstract or paper with collaboration in the last 48 months (publication date may be later); and", null, null, null, null, null, null, null],
  ["C: Collaborators on projects, such as funded awards, graduate research or others in the last 48 months.", null, null, null, null, null, null, null],
  [null, null, null, "To disambiguate common names", null, null, null, null],
  ["4", "Name:", "Organizational Affiliation", "Optional  (email, Department)", "Last Active", null, null, null],
  ["A:", "Narayanan, A.", null, null, null, null, null, null],
  ["A:", "Bunting, E. L.", null, null, null, null, null, null],
  ["A:", "Keellings, David", "University of Florida", "Department of Geography", null, null, null, null],
  ["A:", "Fouladi-Fard, R.", null, null, null, null, null, null],
  ["A:", "O'Shaughnessy, P.", null, null, null, null, null, null],
  ["A:", "Naddafi, K.", null, null, null, null, null, null],
  ["A:", "Karimi, A.", "Qom University of Technology", null, null, null, null, null],
  ["A:", "Jahangir, M. S.", null, null, null, null, null, null],
  ["A:", "Quilty, J.", null, null, null, null, null, null],
  ["A:", "Mojarad, H.", null, null, null, null, null, null],
  ["A:", "Sorooshian, A.", null, null, null, null, null, null],
  ["A:", "Mahdinia, M.", null, null, null, null, null, null],
  ["A:", "Farajollahi, M.", null, null, null, null, null, null],
  ["A:", "Fahiminia, M.", null, null, null, null, null, null],
  ["A:", "Rahimi, N. R.", null, null, null, null, null, null],
  ["A:", "Aali, R.", null, null, null, null, null, null],
  ["A:", "Shahryari, A.", null, null, null, null, null, null],
  ["C:", "Li, Shawn", "Columbia University", "Research project supervisor listed in CV", null, null, null, null],
  [null, null, null, null, null, null, null, null],

  ["Table 5: List editorial board, editor-in chief and co-editors with whom the individual interacts. An editor-in-chief must list the entire editorial board.", null, null, null, null, null, null, null],
  ["B: Editorial Board: List name(s) of editor-in-chief and journal in the past 24 months; and", null, null, null, null, null, null, null],
  ["E: Other co-Editors of journal or collections with whom the individual has directly interacted in the last 24 months.", null, null, null, null, null, null, null],
  [null, null, null, "To disambiguate common names", null, null, null, null],
  ["5", "Name:", "Journal / Collection", "Optional  (email, Department)", "Last Active", null, null, null],
  [null, null, null, null, null, null, null, null],
];

sheet.getRange(`A17:H${16 + rows.length}`).values = rows;
sheet.getRange("A17:H95").format = {
  font: { name: "Arial", size: 10, color: "#000000" },
  wrapText: true,
  verticalAlignment: "top",
};
sheet.getRange("A17:E95").format.borders = { preset: "inside", style: "thin", color: "#D9D9D9" };
sheet.getRange("A17:E95").format.borders = { preset: "outside", style: "thin", color: "#A6A6A6" };

const populated = await workbook.inspect({
  kind: "table",
  range: "NSF COA Template!A15:E65",
  include: "values",
  tableMaxRows: 55,
  tableMaxCols: 5,
  summary: "populated COA",
});
console.log(populated.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|Alphaman|Test University",
  options: { useRegex: true, maxResults: 100 },
  summary: "final error and placeholder scan",
});
console.log(errors.ndjson);

await workbook.render({ sheetName: "NSF COA Template", range: "A15:E65", scale: 1.5 });

await fs.mkdir(outDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
