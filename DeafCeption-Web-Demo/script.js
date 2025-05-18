const video = document.getElementById('video');
const label = document.getElementById('label');
const predictBtn = document.getElementById('predictBtn');
const socket = new WebSocket("wss://dfc-web-demo.onrender.com/ws"); // keep this

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

predictBtn.onclick = () => {
  label.textContent = "Predicting...";

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result.split(',')[1];
      socket.send(base64data);
    };
    reader.readAsDataURL(blob);
  }, 'image/jpeg');
};

socket.onmessage = event => {
  label.textContent = event.data;
};

socket.onerror = err => {
  console.error("WebSocket error:", err);
  label.textContent = "WebSocket Error";
};
