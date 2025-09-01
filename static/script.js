
// Loan Approval Decision Support System - JavaScript (Indian Customized)

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Add form validation
    setupFormValidation();

    // Add interactive elements
    setupInteractivity();

    // Add smooth scrolling
    setupSmoothScrolling();

    // Add loading states
    setupLoadingStates();

    // Setup Indian specific features
    setupIndianFeatures();

    console.log('LoanAI India System initialized successfully');
}

function setupIndianFeatures() {
    // Format Indian currency inputs
    setupCurrencyFormatting();

    // Add loan term calculator
    setupLoanTermCalculator();

    // Add application number refresh
    setupApplicationNumberRefresh();

    // Add Indian validation rules
    setupIndianValidation();
}

function setupCurrencyFormatting() {
    const currencyFields = ['applicant_income', 'coapplicant_income', 'loan_amount'];

    currencyFields.forEach(fieldName => {
        const field = document.getElementById(fieldName);
        if (!field) return;

        // Format on blur
        field.addEventListener('blur', function() {
            const value = parseInt(this.value);
            if (value && value > 0) {
                // Add Indian number formatting info
                const formatted = formatIndianCurrency(value);
                showCurrencyInfo(field, formatted);
            }
        });

        // Real-time validation
        field.addEventListener('input', function() {
            validateIndianCurrency(this);
        });
    });
}

function formatIndianCurrency(amount) {
    // Indian number system formatting
    if (amount >= 10000000) { // 1 crore
        return `₹${(amount/10000000).toFixed(2)} Crores`;
    } else if (amount >= 100000) { // 1 lakh
        return `₹${(amount/100000).toFixed(2)} Lakhs`;
    } else {
        return `₹${amount.toLocaleString('en-IN')}`;
    }
}

function showCurrencyInfo(field, formatted) {
    // Remove existing info
    const existingInfo = field.parentNode.querySelector('.currency-info');
    if (existingInfo) {
        existingInfo.remove();
    }

    // Add formatted info
    const infoDiv = document.createElement('div');
    infoDiv.className = 'currency-info';
    infoDiv.style.cssText = `
        font-size: 0.875rem;
        color: #10b981;
        margin-top: 0.25rem;
        font-weight: 500;
    `;
    infoDiv.textContent = formatted;
    field.parentNode.appendChild(infoDiv);
}

function setupLoanTermCalculator() {
    const loanTermField = document.getElementById('loan_term');
    if (!loanTermField) return;

    loanTermField.addEventListener('input', function() {
        const days = parseInt(this.value);
        if (days && days > 0) {
            const years = (days / 365).toFixed(1);
            const months = Math.round(days / 30.44);

            // Show conversion info
            let infoDiv = this.parentNode.querySelector('.term-info');
            if (!infoDiv) {
                infoDiv = document.createElement('div');
                infoDiv.className = 'term-info';
                infoDiv.style.cssText = `
                    font-size: 0.875rem;
                    color: #667eea;
                    margin-top: 0.25rem;
                    font-weight: 500;
                `;
                this.parentNode.appendChild(infoDiv);
            }

            infoDiv.textContent = `Approximately ${years} years (${months} months)`;
        }
    });
}

function setupApplicationNumberRefresh() {
    const appNumberField = document.getElementById('application_number');
    if (!appNumberField) return;

    // Add refresh button
    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.innerHTML = '<i class="fas fa-refresh"></i>';
    refreshBtn.className = 'refresh-app-number';
    refreshBtn.style.cssText = `
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        background: #667eea;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px;
        cursor: pointer;
        font-size: 12px;
    `;

    // Make parent relative
    appNumberField.parentNode.style.position = 'relative';
    appNumberField.parentNode.appendChild(refreshBtn);

    refreshBtn.addEventListener('click', function() {
        generateNewApplicationNumber();
    });
}

function generateNewApplicationNumber() {
    const year = new Date().getFullYear();
    const randomDigits = Math.floor(Math.random() * 900000) + 100000;
    const newAppNumber = `LA${year}${randomDigits}`;

    const appNumberField = document.getElementById('application_number');
    if (appNumberField) {
        appNumberField.value = newAppNumber;

        // Show success feedback
        showMessage('New application number generated!', 'success');
    }
}

function setupIndianValidation() {
    // Minimum income validation for India
    const incomeField = document.getElementById('applicant_income');
    if (incomeField) {
        incomeField.addEventListener('blur', function() {
            const income = parseInt(this.value);
            if (income && income < 15000) {
                showFieldError(this, 'Minimum income should be ₹15,000 for loan eligibility');
            } else if (income && income > 10000000) {
                showFieldError(this, 'Please contact branch for high-value applications');
            }
        });
    }

    // Loan amount validation
    const loanField = document.getElementById('loan_amount');
    if (loanField) {
        loanField.addEventListener('blur', function() {
            const loan = parseInt(this.value);
            const income = parseInt(document.getElementById('applicant_income').value) || 0;
            const coIncome = parseInt(document.getElementById('coapplicant_income').value) || 0;
            const totalIncome = income + coIncome;

            if (loan && totalIncome && loan > (totalIncome * 12 * 5)) {
                showFieldError(this, 'Loan amount is high compared to annual income (>5x)');
            }
        });
    }

    // Loan term validation
    const termField = document.getElementById('loan_term');
    if (termField) {
        termField.addEventListener('blur', function() {
            const days = parseInt(this.value);
            if (days && days < 365) {
                showFieldError(this, 'Minimum loan term should be 1 year (365 days)');
            } else if (days && days > 14600) { // 40 years
                showFieldError(this, 'Maximum loan term is 40 years (14,600 days)');
            }
        });
    }
}

function validateIndianCurrency(field) {
    const value = parseInt(field.value);

    // Remove non-numeric characters except for necessary ones
    field.value = field.value.replace(/[^0-9]/g, '');

    // Field-specific validation
    if (field.name === 'applicant_income') {
        if (value && (value < 10000 || value > 10000000)) {
            field.style.borderColor = '#f59e0b';
        } else {
            field.style.borderColor = '#10b981';
        }
    } else if (field.name === 'loan_amount') {
        if (value && (value < 100000 || value > 100000000)) {
            field.style.borderColor = '#f59e0b';
        } else {
            field.style.borderColor = '#10b981';
        }
    }
}

function setupFormValidation() {
    const form = document.querySelector('.loan-form');
    if (!form) return;

    const inputs = form.querySelectorAll('input[required], select[required]');

    // Add real-time validation
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearErrors);
    });

    // Form submission validation
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            showFormErrors();
        } else {
            showLoadingState();
        }
    });
}

function validateField(event) {
    const field = event.target;
    const value = field.value.trim();

    // Remove existing error styling
    field.classList.remove('error');

    // Validate based on field type
    let isValid = true;
    let errorMessage = '';

    if (field.hasAttribute('required') && !value) {
        isValid = false;
        errorMessage = 'This field is required';
    } else if (field.type === 'number') {
        const numValue = parseFloat(value);
        const min = parseFloat(field.getAttribute('min')) || 0;
        const max = parseFloat(field.getAttribute('max'));

        if (isNaN(numValue) || numValue < min) {
            isValid = false;
            errorMessage = `Please enter a valid number (minimum: ${min.toLocaleString('en-IN')})`;
        } else if (max && numValue > max) {
            isValid = false;
            errorMessage = `Please enter a number less than ${max.toLocaleString('en-IN')}`;
        }
    } else if (field.type === 'text' && field.name === 'applicant_name') {
        if (value.length < 2) {
            isValid = false;
            errorMessage = 'Please enter a valid full name';
        }
    }

    if (!isValid) {
        showFieldError(field, errorMessage);
    } else {
        showFieldSuccess(field);
    }

    return isValid;
}

function clearErrors(event) {
    const field = event.target;
    field.classList.remove('error', 'success');
    removeFieldMessage(field);
}

function showFieldError(field, message) {
    field.classList.add('error');
    field.classList.remove('success');

    // Remove existing message
    removeFieldMessage(field);

    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.textContent = message;
    errorDiv.style.cssText = `
        color: #ef4444;
        font-size: 0.875rem;
        margin-top: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    `;

    field.parentNode.appendChild(errorDiv);
}

function showFieldSuccess(field) {
    field.classList.add('success');
    field.classList.remove('error');
    removeFieldMessage(field);
}

function removeFieldMessage(field) {
    const messages = field.parentNode.querySelectorAll('.field-error, .field-success, .currency-info, .term-info');
    messages.forEach(msg => {
        if (!msg.classList.contains('field-hint')) {
            msg.remove();
        }
    });
}

function validateForm() {
    const form = document.querySelector('.loan-form');
    const requiredFields = form.querySelectorAll('input[required], select[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!validateField({ target: field })) {
            isValid = false;
        }
    });

    // Additional Indian banking validation
    const income = parseInt(document.getElementById('applicant_income').value) || 0;
    const coIncome = parseInt(document.getElementById('coapplicant_income').value) || 0;
    const loanAmount = parseInt(document.getElementById('loan_amount').value) || 0;
    const totalIncome = income + coIncome;

    if (loanAmount > 0 && totalIncome > 0) {
        const ratio = loanAmount / (totalIncome * 12);
        if (ratio > 6) {
            showMessage('Warning: Loan amount is very high compared to annual income (>6x). This may significantly reduce approval chances.', 'warning');
        }
    }

    return isValid;
}

function setupInteractivity() {
    // Enhanced interactivity for Indian features

    // Add hover effects to cards
    const cards = document.querySelectorAll('.feature-card, .about-card, .developer-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.transition = 'transform 0.3s ease';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Add click effect to buttons
    const buttons = document.querySelectorAll('.submit-btn, .btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
}

function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function setupLoadingStates() {
    const form = document.querySelector('.loan-form');
    if (!form) return;

    form.addEventListener('submit', showLoadingState);
}

function showLoadingState() {
    const submitBtn = document.querySelector('.submit-btn');
    if (!submitBtn) return;

    const originalText = submitBtn.innerHTML;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `
        <span class="loading"></span>
        Processing Application...
    `;

    submitBtn.classList.add('loading-state');

    // Restore button after timeout
    setTimeout(() => {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
        submitBtn.classList.remove('loading-state');
    }, 30000);
}

function showFormErrors() {
    showMessage('Please fill in all required fields correctly.', 'error');
}

function showMessage(message, type = 'info') {
    // Remove existing message
    const existingMessage = document.querySelector('.form-message');
    if (existingMessage) {
        existingMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `form-message ${type}`;

    const icon = type === 'error' ? 'fa-exclamation-circle' : 
                type === 'warning' ? 'fa-exclamation-triangle' : 
                type === 'success' ? 'fa-check-circle' : 'fa-info-circle';

    messageDiv.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;

    const colors = {
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6',
        success: '#10b981'
    };

    messageDiv.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background: ${colors[type]}15;
        border: 1px solid ${colors[type]}40;
        border-radius: 8px;
        color: ${colors[type]};
        font-weight: 500;
        animation: slideIn 0.3s ease;
    `;

    const form = document.querySelector('.loan-form');
    form.insertBefore(messageDiv, form.firstChild);

    // Auto remove after 8 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => messageDiv.remove(), 300);
        }
    }, 8000);
}

// Utility functions
function formatCurrency(amount, currency = 'INR') {
    if (currency === 'INR') {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR'
        }).format(amount);
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-IN').format(num);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }

    .loading {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .field-error {
        animation: slideIn 0.3s ease;
    }

    .form-group input.error,
    .form-group select.error {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
    }

    .form-group input.success,
    .form-group select.success {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
`;
document.head.appendChild(style);

// Initialize auto-save functionality (optional)
function setupAutoSave() {
    const form = document.querySelector('.loan-form');
    if (!form) return;

    const formData = {};

    form.addEventListener('input', function(e) {
        if (e.target.name && e.target.name !== 'application_number') {
            formData[e.target.name] = e.target.value;
            localStorage.setItem('loanFormDataIndia', JSON.stringify(formData));
        }
    });

    // Restore form data on page load
    const savedData = localStorage.getItem('loanFormDataIndia');
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            Object.keys(data).forEach(key => {
                const field = form.querySelector(`[name="${key}"]`);
                if (field && key !== 'application_number') {
                    field.value = data[key];
                }
            });
        } catch (e) {
            console.warn('Could not restore form data:', e);
        }
    }
}

// Clear saved data when form is submitted successfully
function clearSavedData() {
    localStorage.removeItem('loanFormDataIndia');
}

// Performance monitoring
function monitorPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            const loadTime = performance.now();
            console.log(`LoanAI India page loaded in ${loadTime.toFixed(2)}ms`);
        });
    }
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    setupAutoSave();
    monitorPerformance();
});
