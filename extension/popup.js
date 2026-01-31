// Popup.js - Extension popup UI logic

document.addEventListener('DOMContentLoaded', function() {
  console.log('Airbnb Price Predictor popup loaded');

  // Check backend health status
  checkBackendHealth();

  // Re-check every 5 seconds while popup is open
  setInterval(checkBackendHealth, 5000);
});

function checkBackendHealth() {
  chrome.runtime.sendMessage({ action: 'checkHealth' }, (response) => {
    const dot = document.getElementById('backend-dot');
    const status = document.getElementById('backend-status');

    if (chrome.runtime.lastError) {
      console.warn('Health check failed (runtime error):', chrome.runtime.lastError);
      updateStatus(dot, status, false);
      return;
    }

    if (response && response.status === 'connected') {
      updateStatus(dot, status, true);
    } else {
      updateStatus(dot, status, false);
    }
  });
}

function updateStatus(dotElement, statusElement, isConnected) {
  if (isConnected) {
    dotElement.classList.remove('disconnected');
    dotElement.classList.add('connected');
    statusElement.textContent = 'Connected';
    statusElement.style.color = '#10b981';
  } else {
    dotElement.classList.remove('connected');
    dotElement.classList.add('disconnected');
    statusElement.textContent = 'Disconnected';
    statusElement.style.color = '#ef4444';
  }
}
