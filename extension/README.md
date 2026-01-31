# Airbnb Price Stoplight

A Chrome Extension that provides intelligent pricing insights for Airbnb travelers.

## Features

- **Dynamic Date Pricing**: Color-coded stoplights (Green/Yellow/Red) on the calendar to highlight the best travel dates.
- **Property Valuation**: Instant "Overvalued" / "Fair" / "Undervalued" assessment on property detail pages.

## Installation

1. Clone the repository.
2. Start the backend server:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python init_db.py  # Initialize mock data
   python app.py      # Start server
   ```
3. Load the extension in Chrome:
   - Go to `chrome://extensions`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `extension/` directory

## Usage

1. Navigate to [Airbnb](https://www.airbnb.com).
2. Enter a destination and click the "Check-in" date field.
   - **Result**: Visible dates will show colored indicators based on relative price.
3. Click on a property to view details.
   - **Result**: A "Price Prediction" widget will appear (if property ID is in mock DB, e.g., `123456`).

## Testing

### Frontend (Extension)

Unit tests are implemented using Jest.

```bash
npm install
npm test
```

### Backend (API)

Integration tests are implemented using Pytest.

```bash
cd backend
source venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Mock Data

The backend uses a local SQLite database (`backend/database.db`) with mock data. To add more properties:
1. Edit `backend/init_db.py`
2. Run `python backend/init_db.py` to reset and repopulate.