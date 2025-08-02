// Global variables
let currentData = null;
let analysisResults = null;
// Global chart variables
let severityChart = null;
let typeChart = null;
let heatmapChart = null;
let timelineChart = null;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize the application
function initializeApp() {
    setupFileUpload();
    setupChartResize();
}

// File upload functionality
function setupFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (!uploadArea || !fileInput) {
        console.error('Upload elements not found');
        return;
    }

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Handle file upload
function handleFileUpload(file) {
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json'];
    
    if (!allowedTypes.includes(file.type)) {
        showNotification('Please upload a valid file (CSV, Excel, or JSON)', 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) { // 50MB limit
        showNotification('File size must be less than 50MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = parseFileData(e.target.result, file.type);
            currentData = data;
            showNotification('File uploaded successfully! Starting analysis...', 'success');
            analyzeData();
        } catch (error) {
            showNotification('Error parsing file: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);
}

// Parse file data
function parseFileData(content, fileType) {
    if (fileType === 'application/json') {
        return JSON.parse(content);
    } else if (fileType === 'text/csv') {
        return parseCSV(content);
    } else {
        // For Excel files, we'll use a simple CSV parser for demo
        return parseCSV(content);
    }
}

// Parse CSV data
function parseCSV(content) {
    const lines = content.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim()) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index] || '';
            });
            data.push(row);
        }
    }
    
    return data;
}

// Analyze data
function analyzeData() {
    if (!currentData || currentData.length === 0) {
        showNotification('No data to analyze', 'error');
        return;
    }

    // Show analysis section
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('analysis-section').style.display = 'block';

    // Simulate analysis delay
    setTimeout(() => {
        const results = performAnalysis(currentData);
        displayAnalysisResults(results);
        
        // Hide analysis section and show results
        document.getElementById('analysis-section').style.display = 'none';
        document.getElementById('results-section').style.display = 'block';
    }, 3000);
}

// Perform analysis
function performAnalysis(data) {
    // Generate sample analysis results
    const issues = [];
    const totalAmount = data.reduce((sum, row) => {
        const amount = parseFloat(row.Amount || row.amount || 0);
        return sum + Math.abs(amount);
    }, 0);

    // Generate sample issues
    const issueTypes = ['Duplicate Transaction', 'Failed Payment', 'Refund Anomaly', 'Billing Error', 'Customer Churn'];
    const severities = ['Critical', 'High', 'Medium', 'Low'];
    const customers = [...new Set(data.map(row => row.Customer || row.customer || 'Unknown'))];

    for (let i = 0; i < Math.min(8, data.length); i++) {
        const row = data[i];
        const issueType = issueTypes[Math.floor(Math.random() * issueTypes.length)];
        const severity = severities[Math.floor(Math.random() * severities.length)];
        const amount = parseFloat(row.Amount || row.amount || 0) * (0.1 + Math.random() * 0.3);
        
        issues.push({
            type: issueType,
            severity: severity,
            amount: Math.round(amount * 100) / 100,
            date: row.Date || row.date || '2024-01-01',
            customer: row.Customer || row.customer || 'Unknown',
            category: row.Category || row.category || 'General',
            description: `Detected ${issueType.toLowerCase()} for customer ${row.Customer || row.customer || 'Unknown'}`,
            recommendations: [
                'Review transaction details',
                'Contact customer for clarification',
                'Update billing procedures'
            ]
        });
    }

    const criticalIssues = issues.filter(issue => issue.severity === 'Critical').length;
    const mediumIssues = issues.filter(issue => issue.severity === 'Medium').length;
    const lowIssues = issues.filter(issue => issue.severity === 'Low').length;

    return {
        issues: issues,
        totalPotentialLoss: issues.reduce((sum, issue) => sum + issue.amount, 0),
        criticalIssues: criticalIssues,
        mediumIssues: mediumIssues,
        lowIssues: lowIssues
    };
}

// Display analysis results
function displayAnalysisResults(results) {
    // Store results globally for download functions
    window.currentAnalysisResults = results;
    
    const resultsContainer = document.getElementById('analysis-results');
    resultsContainer.innerHTML = '';
    
    // Add download report button
    const downloadSection = document.createElement('div');
    downloadSection.className = 'download-section';
    downloadSection.innerHTML = `
        <div class="download-header">
            <h3>📊 Download Analysis Report</h3>
            <p>Get your complete revenue leakage analysis in multiple formats</p>
        </div>
        <div class="download-buttons">
            <button onclick="downloadReport('pdf')" class="download-btn pdf-btn">
                📄 Download PDF Report
            </button>
            <button onclick="downloadReport('excel')" class="download-btn excel-btn">
                📊 Download Excel Report
            </button>
            <button onclick="downloadReport('csv')" class="download-btn csv-btn">
                📋 Download CSV Data
            </button>
        </div>
    `;
    resultsContainer.appendChild(downloadSection);

    // Summary section
    const summarySection = document.createElement('div');
    summarySection.className = 'summary-section';
    summarySection.innerHTML = `
        <div class="summary-card">
            <h3>📈 Analysis Summary</h3>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">${results.issues.length}</span>
                    <span class="stat-label">Total Issues Found</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">$${results.totalPotentialLoss.toLocaleString()}</span>
                    <span class="stat-label">Potential Revenue Loss</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${results.criticalIssues}</span>
                    <span class="stat-label">Critical Issues</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${results.mediumIssues}</span>
                    <span class="stat-label">Medium Priority</span>
                </div>
            </div>
        </div>
    `;
    resultsContainer.appendChild(summarySection);

    // Enhanced charts section
    const chartsSection = document.createElement('div');
    chartsSection.className = 'charts-section';
    chartsSection.innerHTML = `
        <h3>📊 Advanced Analytics</h3>
        <div class="charts-container">
            <div class="chart-card">
                <h4>Issue Severity Distribution</h4>
                <canvas id="severityChart"></canvas>
            </div>
            <div class="chart-card">
                <h4>Issue Type Breakdown</h4>
                <canvas id="typeChart"></canvas>
            </div>
            <div class="chart-card">
                <h4>Revenue Loss Timeline</h4>
                <canvas id="timelineChart"></canvas>
            </div>
            <div class="chart-card">
                <h4>Issue Heatmap</h4>
                <canvas id="heatmapChart"></canvas>
            </div>
        </div>
    `;
    resultsContainer.appendChild(chartsSection);

    // Detailed issues section
    const issuesSection = document.createElement('div');
    issuesSection.className = 'issues-section';
    issuesSection.innerHTML = `
        <h3>🔍 Detailed Issues</h3>
        <div class="issues-list">
            ${results.issues.map((issue, index) => `
                <div class="issue-card ${issue.severity.toLowerCase()}">
                    <div class="issue-header">
                        <span class="issue-severity ${issue.severity.toLowerCase()}">${issue.severity}</span>
                        <span class="issue-amount">$${issue.amount.toLocaleString()}</span>
                    </div>
                    <h4>${issue.type}</h4>
                    <p>${issue.description}</p>
                    <div class="issue-details">
                        <span class="detail-item">📅 Date: ${issue.date}</span>
                        <span class="detail-item">👤 Customer: ${issue.customer}</span>
                        <span class="detail-item">🏷️ Category: ${issue.category}</span>
                    </div>
                    <div class="issue-recommendations">
                        <strong>💡 Recommendations:</strong>
                        <ul>
                            ${issue.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    resultsContainer.appendChild(issuesSection);

    // Create enhanced charts with delay
    setTimeout(() => {
        createSeverityChart(results.issues);
        createTypeChart(results.issues);
        createTimelineChart(results.issues);
        createHeatmapChart(results.issues);
    }, 100);
}

// Create severity chart
function createSeverityChart(issues) {
    if (severityChart) { severityChart.destroy(); }
    
    const ctx = document.getElementById('severityChart').getContext('2d');
    
    const severityCounts = {};
    issues.forEach(issue => {
        severityCounts[issue.severity] = (severityCounts[issue.severity] || 0) + 1;
    });
    
    severityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(severityCounts),
            datasets: [{
                data: Object.values(severityCounts),
                backgroundColor: [
                    '#DC2626', // Critical - Red
                    '#EA580C', // High - Orange
                    '#D97706', // Medium - Yellow
                    '#059669'  // Low - Green
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Create type chart
function createTypeChart(issues) {
    if (typeChart) { typeChart.destroy(); }
    
    const ctx = document.getElementById('typeChart').getContext('2d');
    
    const typeCounts = {};
    issues.forEach(issue => {
        typeCounts[issue.type] = (typeCounts[issue.type] || 0) + 1;
    });
    
    typeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(typeCounts),
            datasets: [{
                label: 'Number of Issues',
                data: Object.values(typeCounts),
                backgroundColor: '#3B82F6',
                borderColor: '#2563EB',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Create timeline chart
function createTimelineChart(issues) {
    if (timelineChart) { timelineChart.destroy(); }
    
    const ctx = document.getElementById('timelineChart').getContext('2d');
    
    // Group issues by date
    const dateGroups = {};
    issues.forEach(issue => {
        const date = issue.date;
        if (!dateGroups[date]) {
            dateGroups[date] = { total: 0, count: 0 };
        }
        dateGroups[date].total += issue.amount;
        dateGroups[date].count += 1;
    });
    
    const dates = Object.keys(dateGroups).sort();
    const amounts = dates.map(date => dateGroups[date].total);
    const counts = dates.map(date => dateGroups[date].count);
    
    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Revenue Loss ($)',
                data: amounts,
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                yAxisID: 'y'
            }, {
                label: 'Number of Issues',
                data: counts,
                borderColor: '#EF4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Revenue Loss Timeline',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Revenue Loss ($)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Number of Issues'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// Create heatmap chart
function createHeatmapChart(issues) {
    if (heatmapChart) { heatmapChart.destroy(); }
    
    const ctx = document.getElementById('heatmapChart').getContext('2d');
    
    // Create heatmap data by severity and type
    const severityTypes = ['Critical', 'High', 'Medium', 'Low'];
    const issueTypes = [...new Set(issues.map(issue => issue.type))];
    
    const heatmapData = severityTypes.map(severity => 
        issueTypes.map(type => {
            const matchingIssues = issues.filter(issue => 
                issue.severity === severity && issue.type === type
            );
            return matchingIssues.reduce((sum, issue) => sum + issue.amount, 0);
        })
    );
    
    heatmapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: issueTypes,
            datasets: severityTypes.map((severity, index) => ({
                label: severity,
                data: heatmapData[index],
                backgroundColor: getSeverityColor(severity),
                borderColor: getSeverityColor(severity),
                borderWidth: 1
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Revenue Loss by Severity and Type',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Issue Type'
                    }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Revenue Loss ($)'
                    }
                }
            }
        }
    });
}

// Get severity color
function getSeverityColor(severity) {
    const colors = {
        'Critical': '#DC2626',
        'High': '#EA580C',
        'Medium': '#D97706',
        'Low': '#059669'
    };
    return colors[severity] || '#6B7280';
}

// Download report functionality
function downloadReport(format) {
    const results = window.currentAnalysisResults;
    if (!results) {
        alert('No analysis results available for download.');
        return;
    }
    
    switch (format) {
        case 'pdf':
            downloadPDFReport(results);
            break;
        case 'excel':
            downloadExcelReport(results);
            break;
        case 'csv':
            downloadCSVReport(results);
            break;
    }
}

// Download PDF report
function downloadPDFReport(results) {
    try {
        // Create a comprehensive PDF report
        const reportContent = generateReportContent(results);
        
        // Use jsPDF for PDF generation
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Add title
        doc.setFontSize(20);
        doc.text('ZeroLeak.AI - Revenue Leakage Analysis Report', 20, 20);
        
        // Add summary
        doc.setFontSize(14);
        doc.text('Executive Summary', 20, 40);
        doc.setFontSize(10);
        doc.text(`Total Issues Found: ${results.issues.length}`, 20, 50);
        doc.text(`Potential Revenue Loss: $${results.totalPotentialLoss.toLocaleString()}`, 20, 60);
        doc.text(`Critical Issues: ${results.criticalIssues}`, 20, 70);
        doc.text(`Analysis Date: ${new Date().toLocaleDateString()}`, 20, 80);
        
        // Add detailed issues
        let yPosition = 100;
        results.issues.forEach((issue, index) => {
            if (yPosition > 250) {
                doc.addPage();
                yPosition = 20;
            }
            
            doc.setFontSize(12);
            doc.text(`${index + 1}. ${issue.type} - $${issue.amount.toLocaleString()}`, 20, yPosition);
            doc.setFontSize(10);
            doc.text(`Severity: ${issue.severity} | Date: ${issue.date}`, 20, yPosition + 8);
            doc.text(`Description: ${issue.description}`, 20, yPosition + 16);
            yPosition += 30;
        });
        
        // Save the PDF
        doc.save('zeroleak-analysis-report.pdf');
        
        // Show success notification
        showNotification('PDF report downloaded successfully! 📄', 'success');
    } catch (error) {
        console.error('Error generating PDF:', error);
        showNotification('Error generating PDF report. Please try again.', 'error');
    }
}

// Download Excel report
function downloadExcelReport(results) {
    try {
        // Create Excel-like CSV with multiple sheets
        const summaryData = [
            ['Metric', 'Value'],
            ['Total Issues', results.issues.length],
            ['Potential Revenue Loss', `$${results.totalPotentialLoss.toLocaleString()}`],
            ['Critical Issues', results.criticalIssues],
            ['Medium Issues', results.mediumIssues],
            ['Low Issues', results.lowIssues],
            ['Analysis Date', new Date().toLocaleDateString()]
        ];
        
        const issuesData = [
            ['Issue #', 'Type', 'Severity', 'Amount', 'Date', 'Customer', 'Category', 'Description']
        ];
        
        results.issues.forEach((issue, index) => {
            issuesData.push([
                index + 1,
                issue.type,
                issue.severity,
                `$${issue.amount.toLocaleString()}`,
                issue.date,
                issue.customer,
                issue.category,
                issue.description
            ]);
        });
        
        // Create CSV content
        const csvContent = summaryData.map(row => row.join(',')).join('\n') + '\n\n' +
                          issuesData.map(row => row.join(',')).join('\n');
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'zeroleak-analysis-report.csv';
        a.click();
        window.URL.revokeObjectURL(url);
        
        // Show success notification
        showNotification('Excel report downloaded successfully! 📊', 'success');
    } catch (error) {
        console.error('Error generating Excel report:', error);
        showNotification('Error generating Excel report. Please try again.', 'error');
    }
}

// Download CSV report
function downloadCSVReport(results) {
    try {
        // Simple CSV with just the issues data
        const csvContent = [
            ['Type', 'Severity', 'Amount', 'Date', 'Customer', 'Category', 'Description'],
            ...results.issues.map(issue => [
                issue.type,
                issue.severity,
                issue.amount,
                issue.date,
                issue.customer,
                issue.category,
                issue.description
            ])
        ].map(row => row.join(',')).join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'zeroleak-issues.csv';
        a.click();
        window.URL.revokeObjectURL(url);
        
        // Show success notification
        showNotification('CSV data downloaded successfully! 📋', 'success');
    } catch (error) {
        console.error('Error generating CSV report:', error);
        showNotification('Error generating CSV report. Please try again.', 'error');
    }
}

// Generate report content
function generateReportContent(results) {
    return {
        summary: {
            totalIssues: results.issues.length,
            totalLoss: results.totalPotentialLoss,
            criticalIssues: results.criticalIssues,
            mediumIssues: results.mediumIssues,
            lowIssues: results.lowIssues
        },
        issues: results.issues,
        recommendations: generateRecommendations(results.issues)
    };
}

// Generate recommendations
function generateRecommendations(issues) {
    const recommendations = [];
    
    // Analyze patterns and generate recommendations
    const criticalIssues = issues.filter(issue => issue.severity === 'Critical');
    const highValueIssues = issues.filter(issue => issue.amount > 1000);
    
    if (criticalIssues.length > 0) {
        recommendations.push('Immediate action required for critical issues');
    }
    
    if (highValueIssues.length > 0) {
        recommendations.push('Focus on high-value revenue leaks first');
    }
    
    // Add more intelligent recommendations based on patterns
    const issueTypes = issues.map(issue => issue.type);
    const typeCounts = {};
    issueTypes.forEach(type => {
        typeCounts[type] = (typeCounts[type] || 0) + 1;
    });
    
    const mostCommonType = Object.keys(typeCounts).reduce((a, b) => 
        typeCounts[a] > typeCounts[b] ? a : b
    );
    
    recommendations.push(`Most common issue type: ${mostCommonType} - consider process improvements`);
    
    return recommendations;
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Setup chart resize
function setupChartResize() {
    window.addEventListener('resize', function() {
        // Recreate charts on resize if they exist
        if (window.currentAnalysisResults) {
            setTimeout(() => {
                createSeverityChart(window.currentAnalysisResults.issues);
                createTypeChart(window.currentAnalysisResults.issues);
                createTimelineChart(window.currentAnalysisResults.issues);
                createHeatmapChart(window.currentAnalysisResults.issues);
            }, 100);
        }
    });
}

// Modal functions
function showHelp() {
    document.getElementById('help-modal').style.display = 'block';
}

function showAbout() {
    document.getElementById('about-modal').style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

// Close modals when clicking outside
window.addEventListener('click', function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Placeholder functions for footer links
function showPrivacy() {
    showNotification('Privacy Policy - Coming Soon!', 'info');
}

function showTerms() {
    showNotification('Terms of Service - Coming Soon!', 'info');
}

function showContact() {
    showNotification('Contact Us - Coming Soon!', 'info');
} 