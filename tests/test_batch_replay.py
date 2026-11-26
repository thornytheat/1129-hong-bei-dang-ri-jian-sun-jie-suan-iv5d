from bakerylot.pipeline import run


def test_operating_sample_through_receipt_reconcile():
    batch = run(40)
    assert len(batch["stages"]) == 6
    assert len(batch["rows"]) >= 12
    assert any(row["reason"] == "盘亏待核对" for row in batch["rows"])
    assert 0 < batch["metrics"]["预留折扣（元）"] <= 40
    assert batch["metrics"]["待查收据"] >= 1
    assert batch == run(40)


def test_selected_training_window_drives_live_forecast_and_band():
    batch = run(40)
    selection = {row["商品"]: row["训练窗口"]
                 for row in batch["stages"][2]["records"] if "短窗可用日" in row}
    assert selection
    current = {row["lot"]: row["sku"] for row in batch["stages"][4]["records"] if "sku" in row}
    forecasts = [row for row in batch["stages"][2]["records"] if "校准半径" in row]
    assert all(row["训练窗口"] == selection[current[row["lot"]]] for row in forecasts)
    band = {row["商品"]: row for row in batch["stages"][3]["records"]}
    assert all(band[sku]["训练窗口"] == window for sku, window in selection.items())
    short = [row for row in forecasts if row["训练窗口"] == "短窗"]
    assert short and all(row["训练样本"] <= 7 for row in short)


def test_change_allowance_produces_new_audit_record():
    restricted = run(1)
    open_budget = run(100)
    assert len(restricted["audit_id"]) == 64
    assert restricted["audit_id"] != open_budget["audit_id"]
    assert restricted["metrics"]["预留折扣（元）"] < open_budget["metrics"]["预留折扣（元）"]
