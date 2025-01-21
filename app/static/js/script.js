// Prevent form resubmission on refresh
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
}

// Single function to handle image preview
function showImagePreview(imageSource) {
    const imagePreview = document.getElementById('imagePreview');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreviewText = document.getElementById('imagePreviewText');

    // Clear previous preview
    imagePreview.innerHTML = '';

    const img = document.createElement('img');
    img.src = imageSource;
    img.style.maxWidth = '100%';
    img.style.maxHeight = '300px';
    img.style.objectFit = 'contain';
    
    imagePreview.appendChild(img);
    imagePreviewText.style.display = 'block';
    imagePreviewContainer.style.display = 'block';
}

// Handle file input
document.getElementById('file').addEventListener('change', function() {
    const urlInput = document.getElementById('imageUrl');
    urlInput.value = ''; // Clear URL input

    if (this.files && this.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            showImagePreview(e.target.result);
        };
        reader.readAsDataURL(this.files[0]);
    }
});

// Handle URL input
document.getElementById('imageUrl').addEventListener('input', async function() {
    const fileInput = document.getElementById('file');
    fileInput.value = ''; // Clear file input
    
    const url = this.value.trim();
    if (url) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch image');
            
            const blob = await response.blob();
            if (!blob.type.startsWith('image/')) {
                throw new Error('Invalid image format');
            }

            const imageUrl = URL.createObjectURL(blob);
            showImagePreview(imageUrl);
        } catch (error) {
            alert('Error loading image: ' + error.message);
            this.value = '';
        }
    }
});

// Form submission handler
document.getElementById('imageForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file');
    const urlInput = document.getElementById('imageUrl');
    const submitButton = this.querySelector('button[type="submit"]');

    if (!fileInput.files[0] && !urlInput.value.trim()) {
        alert('Please upload an image or provide an image URL');
        return;
    }

    try {
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';

        const formData = new FormData();

        if (fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        } else if (urlInput.value.trim()) {
            const response = await fetch(urlInput.value.trim());
            if (!response.ok) throw new Error('Failed to fetch image');
            
            const blob = await response.blob();
            formData.append('file', blob, 'url_image.jpg');
        }

        const response = await fetch('/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to process image');
        }

        const htmlResponse = await response.text();
        document.documentElement.innerHTML = htmlResponse;

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Search';
    }
});