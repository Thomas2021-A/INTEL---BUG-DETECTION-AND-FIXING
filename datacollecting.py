import os
import pandas as pd


def extract_buggy_fixed_code(patch_lines):
    buggy_code = []
    fixed_code = []

    for line in patch_lines:
        if line.startswith("- ") and not line.startswith("---"):
            buggy_code.append(line[2:].strip())
        elif line.startswith("+ ") and not line.startswith("+++"):
            fixed_code.append(line[2:].strip())

    return "\n".join(buggy_code), "\n".join(fixed_code)


def process_bug_patches(dataset_path):
    bug_data = []

    # Iterate through each project in BugsInPy
    for project in os.listdir(dataset_path):
        project_path = os.path.join(dataset_path, project)
        bugs_path = os.path.join(project_path, "bugs")

        if os.path.exists(bugs_path):
            for bug_id in os.listdir(bugs_path):
                bug_folder = os.path.join(bugs_path, bug_id)
                patch_file = os.path.join(bug_folder, "bug_patch.txt")

                if os.path.exists(patch_file):
                    with open(patch_file, "r", encoding="utf-8") as f:
                        patch_content = f.readlines()

                    buggy_code, fixed_code = extract_buggy_fixed_code(patch_content)
                    bug_data.append([project, bug_id, buggy_code, fixed_code])

    # Convert data to DataFrame
    df = pd.DataFrame(bug_data, columns=["Project", "Bug_ID", "Buggy_Code", "Fixed_Code"])
    return df


dataset_path = r"C:\Users\12a13\PycharmProjects\BUGGY\BugsInPy"

df_final = process_bug_patches(dataset_path)

csv_path = os.path.join(dataset_path, "final_fixed_bug_dataset.csv")
df_final.to_csv(csv_path, index=False)
print(f"Dataset saved at {csv_path}")

xlsx_path = os.path.join(dataset_path, "final_fixed_bug_dataset.xlsx")
df_final.to_excel(xlsx_path, index=False)
print(f"Dataset saved at {xlsx_path}")
