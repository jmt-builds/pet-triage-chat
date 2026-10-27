"""The practice's records: only the signed-in client's animals are ever returned."""

from pet_triage.practice import get_client_patients


def test_only_the_clients_own_patients_are_returned() -> None:
    patients = get_client_patients("client-001")
    assert {patient.name for patient in patients} == {"Bella", "Milo"}
    assert {patient.client_id for patient in patients} == {"client-001"}


def test_an_unknown_client_has_no_patients() -> None:
    assert get_client_patients("nobody") == []
