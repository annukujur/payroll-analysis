"""
Payroll Discrepancy Detection & Compliance Dashboard
Synthetic dataset generator

Generates:
  - employees.csv   : ~500 synthetic employees
  - pay_runs.csv    : ~6,000 pay runs across 6 months, with ~5-8% injected
                       errors and a hidden ground-truth label so you can
                       measure precision/recall of your detection rules later.

Run: python generate_payroll_data.py
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_EMPLOYEES = 500
PAY_FREQUENCIES = ["weekly", "biweekly", "semimonthly"]
FREQ_PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "semimonthly": 24}
STATES = ["CA", "TX", "NY", "FL", "IL", "OH", "GA", "NC", "PA", "WA"]
FILING_STATUS = ["single", "married", "head_of_household"]
GARNISHMENT_TYPES = ["child_support", "tax_levy", "creditor", "none"]

SS_RATE = 0.062
SS_WAGE_BASE_2026 = 176_100  # annual, illustrative
MEDICARE_RATE = 0.0145
MEDICARE_ADDL_RATE = 0.009
MEDICARE_ADDL_THRESHOLD = 200_000
FUTA_WAGE_BASE = 7_000  # per employee per year, employer-side
SUTA_WAGE_BASE = 9_000  # illustrative, varies by state in reality

# Garnishment caps (simplified illustrative rules — CCPA Title III style)
GARNISHMENT_CAP_PCT = {
    "child_support": 0.50,   # up to 50-65% depending on circumstances; using 50% as illustrative cap
    "tax_levy": 0.70,        # IRS levies can be high; illustrative
    "creditor": 0.25,        # standard CCPA cap: 25% of disposable earnings
    "none": 0.0,
}

FEDERAL_TAX_RATE_APPROX = 0.15   # simplified flat approximation for synthetic data
STATE_TAX_RATE_APPROX = {
    "CA": 0.06, "TX": 0.0, "NY": 0.06, "FL": 0.0, "IL": 0.05,
    "OH": 0.04, "GA": 0.05, "NC": 0.05, "PA": 0.03, "WA": 0.0,
}


def generate_employees(n=N_EMPLOYEES):
    rows = []
    for i in range(1, n + 1):
        annual_salary = round(np.random.uniform(32_000, 145_000), 2)
        has_401k = random.random() < 0.55
        garnishment_flag = random.random() < 0.06
        garnishment_type = random.choice(["child_support", "tax_levy", "creditor"]) if garnishment_flag else "none"
        rows.append({
            "employee_id": f"EMP{i:04d}",
            "state": random.choice(STATES),
            "pay_frequency": random.choice(PAY_FREQUENCIES),
            "annual_salary": annual_salary,
            "filing_status": random.choice(FILING_STATUS),
            "exemptions": random.randint(0, 3),
            "has_401k": has_401k,
            "k401_contribution_pct": round(np.random.uniform(0.02, 0.12), 3) if has_401k else 0.0,
            "garnishment_flag": garnishment_flag,
            "garnishment_type": garnishment_type,
            "hire_date": fake.date_between(start_date="-5y", end_date="-30d"),
        })
    return pd.DataFrame(rows)


def compute_clean_pay_run(emp, pay_date, ytd_gross_before):
    """Compute a mathematically correct pay run for one employee/period."""
    periods_per_year = FREQ_PERIODS_PER_YEAR[emp["pay_frequency"]]
    base_gross = emp["annual_salary"] / periods_per_year
    # small natural variance (overtime, bonuses, etc.)
    gross_pay = round(base_gross * np.random.uniform(0.97, 1.05), 2)

    ytd_gross_after = ytd_gross_before + gross_pay

    # Social Security — stop once YTD hits wage base
    ss_taxable = max(0, min(gross_pay, SS_WAGE_BASE_2026 - ytd_gross_before))
    social_security_withheld = round(ss_taxable * SS_RATE, 2)

    # Medicare — no cap, plus additional 0.9% over threshold
    medicare_withheld = round(gross_pay * MEDICARE_RATE, 2)
    if ytd_gross_after > MEDICARE_ADDL_THRESHOLD:
        addl_wages = min(gross_pay, ytd_gross_after - MEDICARE_ADDL_THRESHOLD)
        medicare_withheld += round(max(0, addl_wages) * MEDICARE_ADDL_RATE, 2)

    federal_tax_withheld = round(gross_pay * FEDERAL_TAX_RATE_APPROX, 2)
    state_tax_withheld = round(gross_pay * STATE_TAX_RATE_APPROX[emp["state"]], 2)

    futa_taxable = max(0, min(gross_pay, FUTA_WAGE_BASE - ytd_gross_before))
    futa_withheld = round(futa_taxable * 0.006, 2)
    suta_taxable = max(0, min(gross_pay, SUTA_WAGE_BASE - ytd_gross_before))
    suta_withheld = round(suta_taxable * 0.027, 2)

    k401_contribution = round(gross_pay * emp["k401_contribution_pct"], 2) if emp["has_401k"] else 0.0

    disposable_earnings = gross_pay - federal_tax_withheld - state_tax_withheld - social_security_withheld - medicare_withheld
    garnishment_cap_pct = GARNISHMENT_CAP_PCT.get(emp["garnishment_type"], 0.0)
    garnishment_deduction = round(disposable_earnings * garnishment_cap_pct * np.random.uniform(0.3, 0.9), 2) if emp["garnishment_flag"] else 0.0

    net_pay = round(
        gross_pay - federal_tax_withheld - state_tax_withheld - social_security_withheld
        - medicare_withheld - k401_contribution - garnishment_deduction, 2
    )

    return {
        "gross_pay": gross_pay,
        "federal_tax_withheld": federal_tax_withheld,
        "state_tax_withheld": state_tax_withheld,
        "social_security_withheld": social_security_withheld,
        "medicare_withheld": medicare_withheld,
        "futa_withheld": futa_withheld,
        "suta_withheld": suta_withheld,
        "k401_contribution": k401_contribution,
        "garnishment_deduction": garnishment_deduction,
        "net_pay": net_pay,
        "journal_posted": True,
    }, ytd_gross_after


def inject_error(pay_run, emp):
    """Randomly corrupt a clean pay run and label it. Returns (pay_run, error_type)."""
    error_type = random.choice([
        "tax_miscalculation",
        "missing_401k",
        "garnishment_overwithholding",
        "duplicate_deduction",
        "net_pay_mismatch",
        "late_journal_posting",
        "wage_base_cap_error",
    ])

    if error_type == "tax_miscalculation":
        pay_run["social_security_withheld"] = round(pay_run["social_security_withheld"] * random.choice([0.5, 1.8]), 2)

    elif error_type == "missing_401k":
        if emp["has_401k"]:
            pay_run["k401_contribution"] = 0.0
        else:
            error_type = "tax_miscalculation"
            pay_run["federal_tax_withheld"] = round(pay_run["federal_tax_withheld"] * 1.6, 2)

    elif error_type == "garnishment_overwithholding":
        if emp["garnishment_flag"]:
            pay_run["garnishment_deduction"] = round(pay_run["gross_pay"] * 0.85, 2)  # way past legal cap
        else:
            error_type = "tax_miscalculation"
            pay_run["state_tax_withheld"] = round(pay_run["state_tax_withheld"] * 2.2, 2)

    elif error_type == "duplicate_deduction":
        pay_run["garnishment_deduction"] = round(pay_run["garnishment_deduction"] * 2, 2) if pay_run["garnishment_deduction"] > 0 else round(pay_run["k401_contribution"] * 2, 2)

    elif error_type == "net_pay_mismatch":
        pay_run["net_pay"] = round(pay_run["net_pay"] + np.random.uniform(50, 300) * random.choice([-1, 1]), 2)

    elif error_type == "late_journal_posting":
        pay_run["journal_posted"] = False

    elif error_type == "wage_base_cap_error":
        pay_run["social_security_withheld"] = round(pay_run["gross_pay"] * SS_RATE, 2)  # ignores wage base cap

    return pay_run, error_type


def generate_pay_runs(employees_df, months=6, error_rate=0.06):
    rows = []
    pay_run_counter = 1
    ytd_tracker = {emp_id: 0.0 for emp_id in employees_df["employee_id"]}

    for _, emp in employees_df.iterrows():
        periods_per_year = FREQ_PERIODS_PER_YEAR[emp["pay_frequency"]]
        n_periods = int(periods_per_year * (months / 12))
        pay_dates = pd.date_range(end=pd.Timestamp.today(), periods=n_periods,
                                   freq={"weekly": "W", "biweekly": "2W", "semimonthly": "SME"}[emp["pay_frequency"]])

        for pay_date in pay_dates:
            pay_run, ytd_after = compute_clean_pay_run(emp, pay_date, ytd_tracker[emp["employee_id"]])
            ytd_tracker[emp["employee_id"]] = ytd_after

            is_error = random.random() < error_rate
            error_type = "none"
            if is_error:
                pay_run, error_type = inject_error(pay_run, emp)

            row = {
                "pay_run_id": f"PR{pay_run_counter:06d}",
                "employee_id": emp["employee_id"],
                "pay_date": pay_date.date(),
                **pay_run,
                "is_injected_error": is_error,
                "error_type": error_type,
            }
            rows.append(row)
            pay_run_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating employees...")
    employees_df = generate_employees()
    employees_df.to_csv("employees.csv", index=False)
    print(f"  -> employees.csv ({len(employees_df)} rows)")

    print("Generating pay runs (this may take a moment)...")
    pay_runs_df = generate_pay_runs(employees_df, months=6, error_rate=0.06)
    pay_runs_df.to_csv("pay_runs.csv", index=False)
    print(f"  -> pay_runs.csv ({len(pay_runs_df)} rows)")

    print("\nError breakdown (ground truth):")
    print(pay_runs_df["error_type"].value_counts())

    print(f"\nTotal injected error rate: {pay_runs_df['is_injected_error'].mean():.2%}")
