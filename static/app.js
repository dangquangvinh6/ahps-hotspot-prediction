// Khởi tạo bản đồ
const map = L.map('map').setView([37.0902, -95.7129], 4); 

// Dùng bản đồ OSM Standard để thấy rõ đường và ranh giới
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap',
    maxZoom: 19
}).addTo(map);

let currentMarker = null;

// Tham chiếu UI
const uiCoords = document.getElementById('ui-coords');
const uiState = document.getElementById('ui-state');
const uiCounty = document.getElementById('ui-county');
const featureBody = document.getElementById('feature-body');
const riskCard = document.getElementById('risk-card');
const uiAlert = document.getElementById('ui-alert');
const uiProb = document.getElementById('ui-prob');
const uiProbBar = document.getElementById('ui-prob-bar');
const loadingOverlay = document.getElementById('loading-overlay');

map.on('click', async function(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;

    // Cập nhật marker
    if (currentMarker) map.removeLayer(currentMarker);
    currentMarker = L.marker([lat, lng]).addTo(map);
    
    // UI Loading
    uiCoords.innerText = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    loadingOverlay.classList.remove('hidden');
    resetRiskCard();

    try {
        // 1. GỌI API LẤY FEATURES
        const featRes = await fetch('/extract_features', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lng })
        });
        
        if (!featRes.ok) throw new Error("Feature extraction failed");
        const features = await featRes.json();

        // Cập nhật Location UI
        uiState.innerText = features.Location_State || "Unknown";
        uiCounty.innerText = features.Location_County || "Unknown";

        // Cập nhật Bảng (Bỏ 2 cột Location ra vì đã hiển thị ở trên)
        featureBody.innerHTML = '';
        for (const [key, value] of Object.entries(features)) {
            if (key === 'Location_State' || key === 'Location_County') continue;
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${key}</td><td>${value}</td>`;
            featureBody.appendChild(tr);
        }

        // 2. GỌI API DỰ ĐOÁN (TỰ ĐỘNG)
        const predRes = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        if (!predRes.ok) throw new Error("Prediction failed");
        const result = await predRes.json();

        // 3. HIỂN THỊ KẾT QUẢ VỚI ANIMATION
        updateRiskCard(result.prediction, result.probability);

    } catch (error) {
        console.error(error);
        alert("Lỗi hệ thống: " + error.message);
    } finally {
        loadingOverlay.classList.add('hidden');
    }
});

function resetRiskCard() {
    riskCard.className = 'risk-card';
    uiAlert.innerText = 'ANALYZING...';
    uiProb.innerText = '0.00%';
    uiProbBar.style.width = '0%';
}

function updateRiskCard(alertLevel, probStr) {
    riskCard.className = 'risk-card'; // reset
    
    // Set giá trị
    uiAlert.innerText = alertLevel;
    uiProb.innerText = probStr;
    uiProbBar.style.width = probStr;

    // Set màu sắc
    if (alertLevel === 'SAFE') {
        riskCard.classList.add('status-safe');
    } else if (alertLevel === 'CAUTION') {
        riskCard.classList.add('status-caution');
    } else if (alertLevel === 'HIGH RISK') {
        riskCard.classList.add('status-high');
    }
}