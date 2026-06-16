"""Unit tests for DetectedItem data transformations."""

from __future__ import annotations

import pytest

from homebox_companion.tools.vision.models import BoundingBox, DetectedItem

# All tests in this module are pure unit tests
pytestmark = pytest.mark.unit


class TestGetExtendedFieldsPayload:
    """Test DetectedItem.get_extended_fields_payload() for update operations."""

    def test_with_data_returns_camelcase_dict(self) -> None:
        """Extended fields should use camelCase keys."""
        item = DetectedItem(
            name="Tool",
            quantity=1,
            manufacturer="DeWalt",
            model_number="DCD771",  # ty: ignore[unknown-argument]
            serial_number="SN12345",  # ty: ignore[unknown-argument]
            purchase_price=99.99,  # ty: ignore[unknown-argument]
            purchase_from="Home Depot",  # ty: ignore[unknown-argument]
            notes="Good condition",
        )

        payload = item.get_extended_fields_payload()

        assert payload is not None
        assert payload["manufacturer"] == "DeWalt"
        assert payload["modelNumber"] == "DCD771"
        assert payload["serialNumber"] == "SN12345"
        assert payload["purchasePrice"] == 99.99
        assert payload["purchaseFrom"] == "Home Depot"
        assert payload["notes"] == "Good condition"

    def test_when_empty_returns_none(self) -> None:
        """No extended fields should return None."""
        item = DetectedItem(name="Basic Item", quantity=1)

        payload = item.get_extended_fields_payload()

        assert payload is None


class TestHasExtendedFields:
    """Test DetectedItem.has_extended_fields() detection."""

    @pytest.mark.parametrize(
        "fields,expected",
        [
            ({}, False),  # Basic item without extended fields
            ({"manufacturer": "Bosch"}, True),  # With manufacturer
            ({"purchase_price": 50.0}, True),  # With positive price
            ({"notes": "Damaged"}, True),  # With notes
            ({"model_number": "ABC123"}, True),  # With model number
            ({"serial_number": "SN12345"}, True),  # With serial number
            ({"purchase_from": "Store"}, True),  # With purchase location
        ],
    )
    def test_has_extended_fields(self, fields: dict, expected: bool) -> None:
        """Item with various extended fields should return appropriate result."""
        item = DetectedItem(name="Tool", quantity=1, **fields)

        assert item.has_extended_fields() is expected


class TestPydanticValidation:
    """Test that DetectedItem validates input correctly via Pydantic."""

    def test_rejects_empty_name(self) -> None:
        """Empty name should be rejected by Pydantic validation."""
        with pytest.raises(ValueError):
            DetectedItem(name="", quantity=1)

    def test_rejects_name_too_long(self) -> None:
        """Name longer than 255 chars should be rejected."""
        with pytest.raises(ValueError):
            DetectedItem(name="x" * 300, quantity=1)

    def test_rejects_description_too_long(self) -> None:
        """Description longer than 1000 chars should be rejected."""
        with pytest.raises(ValueError):
            DetectedItem(name="Item", quantity=1, description="y" * 1500)

    def test_rejects_zero_quantity(self) -> None:
        """Zero quantity should be rejected."""
        with pytest.raises(ValueError):
            DetectedItem(name="Item", quantity=0)

    def test_rejects_zero_price(self) -> None:
        """Zero price should be rejected (must be > 0)."""
        with pytest.raises(ValueError):
            DetectedItem(name="Item", quantity=1, purchase_price=0)  # ty: ignore[unknown-argument]

    def test_accepts_valid_item(self) -> None:
        """Valid item should be accepted."""
        item = DetectedItem(
            name="Valid Item",
            quantity=1,
            description="A valid description",
        )
        assert item.name == "Valid Item"
        assert item.quantity == 1


class TestBoundingBox:
    """Test the optional per-item bounding box used for cropping previews."""

    def test_absent_by_default(self) -> None:
        """An item without a box should have bounding_box None."""
        item = DetectedItem(name="Item", quantity=1)

        assert item.bounding_box is None

    def test_populated_via_camelcase_alias(self) -> None:
        """The LLM returns camelCase boundingBox, which should populate bounding_box."""
        item = DetectedItem.model_validate(
            {"name": "Item", "quantity": 1, "boundingBox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}
        )

        assert item.bounding_box is not None
        assert item.bounding_box.x == 0.1
        assert item.bounding_box.width == 0.3

    def test_out_of_range_values_are_accepted(self) -> None:
        """Out-of-range coordinates must not raise, or a bad box would drop the whole item."""
        item = DetectedItem.model_validate(
            {"name": "Item", "quantity": 1, "boundingBox": {"x": -0.1, "y": 1.5, "width": 1.2, "height": 0.4}}
        )

        assert item.bounding_box is not None
        assert item.bounding_box.y == 1.5

    def test_box_not_in_extended_fields_payload(self) -> None:
        """The box must never leak into the Homebox update payload."""
        item = DetectedItem.model_validate(
            {"name": "Item", "quantity": 1, "boundingBox": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}}
        )

        assert item.get_extended_fields_payload() is None

    def test_model_accepts_plain_floats(self) -> None:
        """BoundingBox is a simple container of normalized floats."""
        box = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)

        assert (box.x, box.y, box.width, box.height) == (0.0, 0.0, 1.0, 1.0)
