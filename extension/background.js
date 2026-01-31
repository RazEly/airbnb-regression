// background.js

// Cache the colors to avoid constant refetching
let colorCache = {
  location: null,
  data: null
};

// Function to fetch colors from the backend
async function fetchColors(location = '') {
  try {
    const url = new URL('http://127.0.0.1:5001/colors');
    if (location) {
      url.searchParams.append('location', location);
    }

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    const data = await response.json();

    // Update cache
    colorCache = {
      location: location,
      data: data
    };

    console.log(`Background: Colors fetched successfully for location "${location}"`, Object.keys(data).length, 'entries');
    return data;
  } catch (error) {
    console.error('Background: Failed to fetch colors', error);
    return null;
  }
}

async function uploadListingDocument(payload = {}) {
  try {
    const listingId = payload?.listingId || payload?.listing_id || 'unknown';
    const htmlLength = payload?.html ? payload.html.length : 0;
    console.log(`Background: Uploading listing ${listingId} (html length: ${htmlLength})`);

    const response = await fetch('http://127.0.0.1:5001/listing', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    const result = await response.json();
    console.log(`Background: Backend acknowledged listing ${listingId} with status ${result?.status || 'unknown'}`);
    return result;
  } catch (error) {
    console.error('Background: Failed to upload listing document', error);
    throw error;
  }
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getColors') {
    const requestedLocation = request.location || '';

    // Check cache
    if (colorCache.data && colorCache.location === requestedLocation) {
      console.log('Background: Returning cached colors for', requestedLocation);
      sendResponse({ colors: colorCache.data });
    } else {
      // Fetch fresh
      fetchColors(requestedLocation).then(data => {
        sendResponse({ colors: data });
      });
      return true; // Indicates async response
    }
  } else if (request.action === 'checkHealth') {
    fetch('http://127.0.0.1:5001/health')
      .then(response => {
        if (response.ok) {
          sendResponse({ status: 'connected' });
        } else {
          sendResponse({ status: 'disconnected' });
        }
      })
      .catch(() => sendResponse({ status: 'disconnected' }));
    return true; // Indicates async response
  } else if (request.action === 'sendListingHtml') {
    uploadListingDocument(request.payload)
      .then((data) => {
        // Pass through the full response including prediction data
        sendResponse(data || { status: 'ok' });
      })
      .catch((error) => {
        sendResponse({ status: 'error', message: error.message });
      });
    return true; // async response
  }
});

