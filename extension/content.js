// Airbnb Calendar Dots Extension
// Adds colored outlines to calendar dates based on data from SQLite backend

(function () {
  'use strict';

  console.log('Airbnb Calendar Dots: Extension v2.5 loaded (Backend scraping relay)');

  const PROCESSED_ATTRIBUTE = 'data-airbnb-color-added';
  const DATE_REGEX = /\d{4}-\d{2}-\d{2}/; // YYYY-MM-DD
  const LISTING_URL_REGEX = /https?:\/\/(?:www\.)?airbnb\.[^/]+\/rooms\//i;
  const LISTING_CHECK_INTERVAL = 1500;
  const LISTING_UPLOAD_DEBOUNCE_MS = 1200;
  const LISTING_RESCRAPE_WINDOW_MS = 60 * 1000; // 1 minute

  // Global map to store date -> color
  let dateColorMap = {};
  let dataLoaded = false;
  let currentLocation = null; // Start as null to force initial fetch
  let isFetching = false;
  let inputListenerAttached = false;
  let fetchTimeout = null;
  let activeInput = null; // Track the specific input the user is using
  let lastFetchedValidLocation = null; // Track the last successfully fetched valid location

  // State
  let isExtensionEnabled = true;
  let isBackendConnected = false;
  let listingWatcherInitialized = false;
  let lastObservedUrl = window.location.href;
  let lastScrapedListingId = null;
  let lastListingScrapeTimestamp = 0;

  // --- UI & Popup Logic ---

  function createPopup() {
    // Avoid duplicates
    if (document.getElementById('airbnb-extension-popup')) return;

    const popup = document.createElement('div');
    popup.id = 'airbnb-extension-popup';
    
    // Status Dot
    const statusDot = document.createElement('span');
    statusDot.className = 'status-dot disconnected'; // Default to disconnected
    statusDot.title = 'Backend Connection Status';
    
    // Label
    const label = document.createElement('span');
    label.textContent = 'Airbnb Colors';
    label.style.fontWeight = '600';

    // Toggle Switch
    const toggleLabel = document.createElement('label');
    toggleLabel.className = 'airbnb-toggle-switch';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = isExtensionEnabled;
    checkbox.addEventListener('change', (e) => {
      toggleExtension(e.target.checked);
    });

    const slider = document.createElement('span');
    slider.className = 'airbnb-slider';

    toggleLabel.appendChild(checkbox);
    toggleLabel.appendChild(slider);

    // Assemble
    popup.appendChild(statusDot);
    popup.appendChild(label);
    popup.appendChild(toggleLabel);

    document.body.appendChild(popup);
  }

  function updateStatusDot(connected) {
    const dot = document.querySelector('#airbnb-extension-popup .status-dot');
    if (!dot) return;

    if (connected) {
      dot.classList.remove('disconnected');
      dot.classList.add('connected');
      dot.title = 'Backend Connected';
    } else {
      dot.classList.remove('connected');
      dot.classList.add('disconnected');
      dot.title = 'Backend Disconnected';
    }
  }

  function checkBackendHealth() {
    chrome.runtime.sendMessage({ action: 'checkHealth' }, (response) => {
      if (chrome.runtime.lastError) {
        console.warn('Health check failed (runtime error):', chrome.runtime.lastError);
        isBackendConnected = false;
      } else if (response && response.status === 'connected') {
        isBackendConnected = true;
      } else {
        isBackendConnected = false;
      }
      updateStatusDot(isBackendConnected);
    });
  }

  function toggleExtension(enabled) {
    isExtensionEnabled = enabled;
    console.log(`Extension toggled: ${enabled ? 'ON' : 'OFF'}`);

    if (enabled) {
      // Refresh connection check
      checkBackendHealth();
      // Trigger processing
      processButtons();
    } else {
      // Clear visual effects
      clearProcessedAttributes();
    }
  }


  // --- Core Logic ---

  // Get location from input
  function getLocation() {
    // 1. If we have a verified active input (user typed in it), trust it first
    if (activeInput && document.body.contains(activeInput)) {
      const value = activeInput.value.trim();
      console.log(`getLocation: Using activeInput, value: "${value}"`);
      return value;
    }

    // 2. Standard Priority search
    const input = document.getElementById('stays-where-input') ||
      document.querySelector('[data-testid="search_query_input"]') ||
      document.querySelector('input[placeholder*="Search destinations"]') ||
      document.querySelector('input[placeholder*="חיפוש יעדים"]');

    // Debugging logic to see what we found
    if (input) {
      const inputValue = input.value.trim();
      console.log(`getLocation: Found input by selector, value: "${inputValue}"`);

      // Attach listener if not already done
      if (!inputListenerAttached) {
        console.log('Attaching input listener to:', input);
        input.addEventListener('input', handleInput);
        input.addEventListener('change', handleInput);
        input.addEventListener('blur', handleInput);
        inputListenerAttached = true;
      }
      return inputValue;
    }

    // 3. Fallback to labels
    const littleSearchLabel = document.querySelector('[data-testid="little-search-label"]');
    if (littleSearchLabel) {
      console.log(`getLocation: Using label fallback, value: "${littleSearchLabel.textContent}"`);
      return littleSearchLabel.textContent;
    }

    console.log('getLocation: No input found, returning empty string');
    return '';
  }

  // Handle input events with debounce
  let debounceTimer;
  function handleInput(e) {
    activeInput = e.target; // Capture this as the authoritative source
    console.log('handleInput triggered. Value:', JSON.stringify(activeInput.value.trim()));

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const currentInput = document.querySelector('[data-testid="search_query_input"]') || 
                           document.getElementById('stays-where-input') ||
                           document.querySelector('input[placeholder*="Search destinations"]') ||
                           document.querySelector('input[placeholder*="חיפוש יעדים"]');
      const location = currentInput ? currentInput.value.trim() : '';
      console.log('handleInput debounce complete. Location:', JSON.stringify(location));
      if (location && location !== lastFetchedValidLocation) {
        console.log('handleInput: Triggering fetch for location:', JSON.stringify(location));
        fetchColors(location);
      } else if (!location) {
        console.log('handleInput: Empty location, not triggering fetch');
      } else {
        console.log('handleInput: Location already fetched:', JSON.stringify(location));
      }
    }, 1000);
  }

  // Fetch colors from background script
  function fetchColors(location) {
    if (!isExtensionEnabled) return; // Don't fetch if disabled

    // Don't fetch for empty/invalid locations
    if (!location || location.trim() === '') {
      console.log(`Skipping fetch for empty location: "${location}"`);
      return;
    }

    if (isFetching) {
      console.log(`Skipping fetch for "${location}" - already fetching.`);
      return;
    }
    isFetching = true;

    // Safety timeout
    clearTimeout(fetchTimeout);
    fetchTimeout = setTimeout(() => {
      if (isFetching) {
        console.warn('Fetch timed out, resetting lock.');
        isFetching = false;
      }
    }, 5000);

    console.log(`Requesting colors from background for location: "${location}"...`);

    chrome.runtime.sendMessage({ action: 'getColors', location: location }, (response) => {
      isFetching = false;
      clearTimeout(fetchTimeout);

      if (chrome.runtime.lastError) {
        console.error('Runtime error:', chrome.runtime.lastError);
        return;
      }

      if (response && response.colors) {
        dateColorMap = response.colors;
        dataLoaded = true;
        currentLocation = location;
        lastFetchedValidLocation = location; // Track this as the last valid location
        console.log(`Received ${Object.keys(dateColorMap).length} color entries for "${location}".`);

        clearProcessedAttributes();
        processButtons();
      } else {
        console.warn('Failed to receive colors or empty response.');
      }
    });
  }

  function clearProcessedAttributes() {
    const processed = document.querySelectorAll(`[${PROCESSED_ATTRIBUTE}]`);
    for (let el of processed) {
      el.removeAttribute(PROCESSED_ATTRIBUTE);
      el.classList.remove('airbnb-day-green', 'airbnb-day-yellow', 'airbnb-day-orange', 'airbnb-day-red');
    }
  }

  function getDateFromElement(element) {
    const extract = (str) => {
      if (!str) return null;
      const match = str.match(DATE_REGEX);
      return match ? match[0] : null;
    };

    const dateState = element.getAttribute('data-state--date-string');
    if (dateState) {
      const date = extract(dateState);
      if (date) return date;
    }

    const label = element.getAttribute('aria-label');
    if (label) {
      const date = extract(label);
      if (date) return date;
    }

    return null;
  }

  // Main function to process buttons
  function processButtons() {
    if (!isExtensionEnabled) return; // Feature flag check

    // 1. Check for location change
    const location = getLocation();

    // Skip fetch for empty locations
    if (!location || location.trim() === '') {
      if (currentLocation !== null) {
        console.log(`processButtons: Location is empty (""), clearing colors.`);
        clearProcessedAttributes();
        dateColorMap = {};
        dataLoaded = false;
        currentLocation = null;
      }
      return;
    }

    // Debug log for decision making
    if (!isFetching && location !== currentLocation) {
      console.log(`Triggering fetch. Reason: New location "${location}" != Old "${currentLocation}".`);
      fetchColors(location);
      return;
    } else if (location !== currentLocation) {
      console.log(`Update pending but fetch locked. New: "${location}", Old: "${currentLocation}", fetching: ${isFetching}`);
    }

    // Initial fetch fallback - only if we haven't fetched a valid location yet
    if (currentLocation === null && !isFetching) {
      fetchColors(location);
      return;
    }

    if (!dataLoaded) return;

    // Optimization: Check for buttons and colored count
    const buttons = document.getElementsByTagName('button');
    const divs = document.querySelectorAll('div[role="button"]');
    const allElements = [...buttons, ...divs];

    for (let element of allElements) {
      if (element.hasAttribute(PROCESSED_ATTRIBUTE)) continue;

      const dateStr = getDateFromElement(element);

      if (dateStr && dateColorMap[dateStr]) {
        const colorClass = dateColorMap[dateStr];
        element.classList.add(colorClass);
        element.setAttribute(PROCESSED_ATTRIBUTE, 'true');
      }
    }
  }

  // Observe DOM changes
  function observeCalendar() {
    const observer = new MutationObserver(() => {
      // Re-run process on DOM changes (e.g. calendar appearing)
      processButtons();
      monitorListingContext();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class', 'data-testid', 'data-state--date-string']
    });
  }

  function start() {
    console.log('Extension starting...');

    createPopup(); // Inject UI
    checkBackendHealth(); // Initial check

    observeCalendar();
    initListingWatcher();
    
    // Try immediate process
    setTimeout(processButtons, 1000);
  }

  function initListingWatcher() {
    if (listingWatcherInitialized) return;
    listingWatcherInitialized = true;

    handleListingUrl(window.location.href);

    setInterval(() => {
      monitorListingContext();
    }, LISTING_CHECK_INTERVAL);
  }

  function monitorListingContext() {
    if (!isExtensionEnabled) return;
    const currentUrl = window.location.href;
    if (currentUrl !== lastObservedUrl) {
      lastObservedUrl = currentUrl;
      
      // Hide prediction overlay when navigating away from listing
      if (!LISTING_URL_REGEX.test(currentUrl) && window.AirbnbPredictionOverlay) {
        window.AirbnbPredictionOverlay.hide();
      }
      
      handleListingUrl(currentUrl);
    } else if (LISTING_URL_REGEX.test(currentUrl) && !lastScrapedListingId) {
      handleListingUrl(currentUrl);
    }
  }

  function handleListingUrl(url) {
    if (!LISTING_URL_REGEX.test(url)) return;
    const listingId = extractListingIdFromUrl(url);
    if (!listingId) return;
    const now = Date.now();
    if (lastScrapedListingId === listingId && now - lastListingScrapeTimestamp < LISTING_RESCRAPE_WINDOW_MS) {
      return;
    }
    lastScrapedListingId = listingId;
    lastListingScrapeTimestamp = now;
    setTimeout(() => uploadListingDocument(listingId, url), LISTING_UPLOAD_DEBOUNCE_MS);
  }

  function extractListingIdFromUrl(url) {
    const match = url.match(/rooms\/([^/?#]+)/i);
    return match ? match[1] : null;
  }

  function uploadListingDocument(listingId, url) {
    if (!isExtensionEnabled) return;
    const payload = buildListingPayload(listingId, url);
    if (!payload.html || payload.html.length < 1000) {
      console.warn('Airbnb Listing Scraper: HTML payload too small to upload.');
      return;
    }

    // Show loading overlay while prediction is being calculated
    if (window.AirbnbPredictionOverlay) {
      window.AirbnbPredictionOverlay.showLoading();
    }

    chrome.runtime.sendMessage({ action: 'sendListingHtml', payload }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('Airbnb Listing Scraper: upload failed', chrome.runtime.lastError);
        if (window.AirbnbPredictionOverlay) {
          window.AirbnbPredictionOverlay.showError('Failed to connect to backend');
        }
        return;
      }
      
      if (response && response.status === 'ok') {
        console.log(`Airbnb Listing Scraper: uploaded listing ${listingId}`);
        
        // Handle prediction response
        if (response.prediction) {
          console.log('Airbnb Prediction: Received prediction data', response.prediction);
          
          if (window.AirbnbPredictionOverlay) {
            if (response.prediction.error) {
              console.log('Airbnb Prediction: Showing error overlay');
              window.AirbnbPredictionOverlay.showError(response.prediction.error);
            } else {
              console.log('Airbnb Prediction: Showing prediction overlay for listing', listingId);
              window.AirbnbPredictionOverlay.updateForListing(listingId, response.prediction);
            }
          }
        } else {
          console.warn('Airbnb Prediction: No prediction data in response - not hiding overlay');
          // Don't hide overlay if there's no prediction, just log warning
          // The loading overlay should remain visible or backend may still be processing
        }
      } else {
        console.warn('Airbnb Listing Scraper: backend did not confirm upload', response);
        if (window.AirbnbPredictionOverlay) {
          window.AirbnbPredictionOverlay.showError('Backend processing failed');
        }
      }
    });
  }

  function buildListingPayload(listingId, url) {
    const html = document.documentElement ? document.documentElement.outerHTML : '';
    const ldJson = Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
      .map((script) => script.textContent)
      .filter(Boolean);

    return {
      listingId,
      url,
      html,
      ldJson,
      capturedAt: new Date().toISOString(),
      title: document.title || null,
      locationHint: getLocation() || null
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  setInterval(() => {
    // Only process periodically if enabled
    if (isExtensionEnabled) {
      processButtons();
    }
  }, 2000);

})();
