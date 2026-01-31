// Airbnb Price Prediction Overlay
// Displays ML price predictions with stoplight indicators on Airbnb listing pages

  (function () {
  'use strict';

  console.log('Airbnb Prediction Overlay: Module loaded');

  let currentOverlay = null;
  let currentListingId = null;
  let overlayLocked = false; // Prevent premature removal

  /**
   * Show prediction overlay with stoplight indicator
   * @param {Object} prediction - Prediction data from backend
   * @param {string} prediction.stoplight - 'good', 'neutral', or 'bad'
   * @param {number} prediction.predicted_price_per_night_usd - Predicted price in USD
   * @param {number} prediction.listed_price_per_night_usd - Listed price in USD
   * @param {number} prediction.predicted_price_original - Predicted price in original currency
   * @param {number} prediction.listed_price_original - Listed price in original currency
   * @param {number} prediction.difference_original - Difference in original currency
   * @param {number} prediction.difference_usd - Dollar difference
   * @param {number} prediction.difference_pct - Percentage difference
   * @param {string} prediction.currency_symbol - Currency symbol (e.g., '$', '€', '₪')
   * @param {string} prediction.city - City name
   * @param {string} prediction.cluster_id - Cluster ID
   */
  function showPredictionOverlay(prediction) {
    console.log('showPredictionOverlay called with:', prediction);
    
    // Remove existing overlay if present
    if (currentOverlay && currentOverlay.parentNode) {
      currentOverlay.parentNode.removeChild(currentOverlay);
      currentOverlay = null;
    }

    if (!prediction || prediction.error) {
      console.warn('Prediction overlay: No valid prediction data', prediction);
      return;
    }

    // Create overlay container
    const overlay = document.createElement('div');
    overlay.id = 'airbnb-prediction-overlay';
    overlay.className = 'airbnb-prediction-overlay';

    // Determine stoplight state
    const stoplight = prediction.stoplight || 'neutral';
    const stoplightClass = `stoplight-${stoplight}`;

    // Get prices - use original currency if available, fallback to USD
    const currencySymbol = prediction.currency_symbol || '$';
    const listed = prediction.listed_price_original || prediction.listed_price_per_night_usd || 0;
    const predicted = prediction.predicted_price_original || prediction.predicted_price_per_night_usd || 0;
    const diffAmount = prediction.difference_original || Math.abs(prediction.difference_usd || 0);
    const diffPct = prediction.difference_pct || 0;
    const city = prediction.city || 'Unknown';
    const cluster = prediction.cluster_id || 'N/A';

    // Calculate savings/overpay
    const savingsAmount = Math.abs(diffAmount);
    const savingsPct = Math.abs(diffPct);

    // Determine label and icon
    let statusLabel, statusIcon, statusMessage;
    if (stoplight === 'good') {
      statusLabel = 'GOOD DEAL';
      statusIcon = '🟢';
      statusMessage = `You save ${currencySymbol}${savingsAmount.toFixed(0)} (${savingsPct.toFixed(0)}%)`;
    } else if (stoplight === 'bad') {
      statusLabel = 'OVERPRICED';
      statusIcon = '🔴';
      statusMessage = `Overpriced by ${currencySymbol}${savingsAmount.toFixed(0)} (${savingsPct.toFixed(0)}%)`;
    } else {
      statusLabel = 'FAIR PRICE';
      statusIcon = '🟡';
      statusMessage = `Within ${savingsPct.toFixed(0)}% of predicted value`;
    }

    // Build overlay HTML
    overlay.innerHTML = `
      <div class="prediction-header ${stoplightClass}">
        <div class="prediction-status">
          <span class="stoplight-icon">${statusIcon}</span>
          <span class="status-label">${statusLabel}</span>
        </div>
        <button class="prediction-close" aria-label="Close prediction">✕</button>
      </div>
      
      <div class="prediction-body">
        <div class="price-comparison">
          <div class="price-row">
            <span class="price-label">Listed:</span>
            <span class="price-value">${currencySymbol}${listed.toFixed(0)}/night</span>
          </div>
          <div class="price-row">
            <span class="price-label">Predicted:</span>
            <span class="price-value">${currencySymbol}${predicted.toFixed(0)}/night</span>
          </div>
          <div class="price-row highlight">
            <span class="price-label">${stoplight === 'good' ? 'You save:' : stoplight === 'bad' ? 'Overpay:' : 'Difference:'}</span>
            <span class="price-value ${stoplightClass}">${statusMessage}</span>
          </div>
        </div>

        <div class="prediction-details">
          <div class="detail-row">
            <span class="detail-icon">📍</span>
            <span class="detail-text">${city}</span>
          </div>
          <div class="detail-row">
            <span class="detail-icon">🗺️</span>
            <span class="detail-text">Cluster ${cluster}</span>
          </div>
        </div>

        <div class="prediction-footer">
        </div>
      </div>
    `;

    // Add close button handler
    const closeBtn = overlay.querySelector('.prediction-close');
    closeBtn.addEventListener('click', () => {
      overlayLocked = false; // Unlock when user manually closes
      hidePredictionOverlay();
    });

    // Add to page
    document.body.appendChild(overlay);
    currentOverlay = overlay;

    // Lock overlay for 2 seconds to prevent premature closing
    overlayLocked = true;
    setTimeout(() => {
      overlayLocked = false;
      console.log('Overlay unlocked, can now be hidden by navigation');
    }, 2000);

    // Animate entrance
    setTimeout(() => {
      overlay.classList.add('visible');
    }, 10);

    console.log('Prediction overlay displayed:', {
      stoplight,
      listed,
      predicted,
      city,
      cluster,
    });
  }

  /**
   * Hide and remove the prediction overlay
   */
  function hidePredictionOverlay() {
    if (overlayLocked) {
      console.log('Overlay is locked, ignoring hide request');
      return;
    }
    
    if (currentOverlay) {
      console.log('Hiding prediction overlay');
      currentOverlay.classList.remove('visible');
      setTimeout(() => {
        if (currentOverlay && currentOverlay.parentNode) {
          currentOverlay.parentNode.removeChild(currentOverlay);
        }
        currentOverlay = null;
        currentListingId = null;
      }, 300); // Wait for fade-out animation
    }
  }

  /**
   * Show loading state while prediction is being calculated
   */
  function showLoadingOverlay() {
    hidePredictionOverlay();

    const overlay = document.createElement('div');
    overlay.id = 'airbnb-prediction-overlay';
    overlay.className = 'airbnb-prediction-overlay prediction-loading';

    overlay.innerHTML = `
      <div class="prediction-header">
        <div class="prediction-status">
          <span class="status-label">Analyzing Price...</span>
        </div>
        <button class="prediction-close" aria-label="Close">✕</button>
      </div>
      
      <div class="prediction-body">
        <div class="loading-spinner">
          <div class="spinner"></div>
          <p>Running ML prediction...</p>
        </div>
      </div>
    `;

    const closeBtn = overlay.querySelector('.prediction-close');
    closeBtn.addEventListener('click', hidePredictionOverlay);

    document.body.appendChild(overlay);
    currentOverlay = overlay;

    setTimeout(() => {
      overlay.classList.add('visible');
    }, 10);
  }

  /**
   * Show error state when prediction fails
   */
  function showErrorOverlay(errorMessage) {
    hidePredictionOverlay();

    const overlay = document.createElement('div');
    overlay.id = 'airbnb-prediction-overlay';
    overlay.className = 'airbnb-prediction-overlay prediction-error';

    overlay.innerHTML = `
      <div class="prediction-header">
        <div class="prediction-status">
          <span class="stoplight-icon">⚠️</span>
          <span class="status-label">Prediction Failed</span>
        </div>
        <button class="prediction-close" aria-label="Close">✕</button>
      </div>
      
      <div class="prediction-body">
        <div class="error-message">
          <p>${errorMessage || 'Unable to calculate price prediction'}</p>
          <p class="error-hint">This may happen if the listing data is incomplete or the location is not in our training dataset.</p>
        </div>
      </div>
    `;

    const closeBtn = overlay.querySelector('.prediction-close');
    closeBtn.addEventListener('click', hidePredictionOverlay);

    document.body.appendChild(overlay);
    currentOverlay = overlay;

    setTimeout(() => {
      overlay.classList.add('visible');
    }, 10);
  }

  /**
   * Update overlay for a specific listing ID (prevent duplicates)
   */
  function updatePredictionForListing(listingId, prediction) {
    // Always remove existing overlay before showing new one
    if (currentOverlay) {
      // Immediate removal without animation for updates
      if (currentOverlay.parentNode) {
        currentOverlay.parentNode.removeChild(currentOverlay);
      }
      currentOverlay = null;
    }
    
    currentListingId = listingId;
    
    if (prediction && prediction.error) {
      showErrorOverlay(prediction.error);
    } else if (prediction) {
      showPredictionOverlay(prediction);
    }
  }

  // Export functions to global scope for use by content.js
  window.AirbnbPredictionOverlay = {
    show: showPredictionOverlay,
    hide: hidePredictionOverlay,
    showLoading: showLoadingOverlay,
    showError: showErrorOverlay,
    updateForListing: updatePredictionForListing,
  };

  console.log('Airbnb Prediction Overlay: Functions exported to window.AirbnbPredictionOverlay');
})();
