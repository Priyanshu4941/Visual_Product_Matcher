// Prevent form resubmission on refresh
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
}

function toggleInputs(type) {
    const fileInput = document.getElementById('file');
    const urlInput = document.getElementById('imageUrl');
    const imagePreviewText = document.getElementById('imagePreviewText');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');

    // Clear previous preview
    imagePreview.innerHTML = '';

    if (type === 'file' && fileInput.files[0]) {
        urlInput.value = '';
        const file = fileInput.files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            fileInput.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '300px';
            img.style.objectFit = 'contain';
            imagePreview.appendChild(img);
            
            imagePreviewText.style.display = 'block';
            imagePreviewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } 
    else if (type === 'url' && urlInput.value) {
        fileInput.value = '';
        const img = document.createElement('img');
        img.src = urlInput.value;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '300px';
        img.style.objectFit = 'contain';
        
        img.onload = function() {
            imagePreview.appendChild(img);
            imagePreviewText.style.display = 'block';
            imagePreviewContainer.style.display = 'block';
        };
        
        img.onerror = function() {
            alert('Invalid image URL or image not found.');
            urlInput.value = '';
        };
    }
}

// Form submission handler
document.getElementById('imageForm').addEventListener('submit', function(event) {
    const fileInput = document.getElementById('file');
    const urlInput = document.getElementById('imageUrl');
    const submitButton = this.querySelector('button[type="submit"]');

    if (!fileInput.files[0] && !urlInput.value.trim()) {
        event.preventDefault();
        alert('Please upload an image or provide an image URL');
        return;
    }

    // Show loading state
    submitButton.disabled = true;
    submitButton.textContent = 'Searching...';
});