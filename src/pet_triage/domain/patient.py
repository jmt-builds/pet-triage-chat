"""The patient: the animal being triaged."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Patient:
    """An animal registered to a client at the practice.

    Read from the practice's records (``pet_triage.practice.get_client_patients``), so the
    owner is never asked for these details, only which of their animals the call is about.

    Attributes:
        id: The practice's ID for the animal.
        client_id: The client the animal is registered to.
        name: What the owner calls it.
        species: The species; only supported species are triaged.
        breed: The breed, which also helps the owner say which animal they mean.
        sex: "male" or "female".
        neutered: Whether the animal is neutered.
        age_years: Age in years.
    """

    id: str
    client_id: str
    name: str
    species: str
    breed: str
    sex: str
    neutered: bool
    age_years: float
