document.getElementById("getEmailButton").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    document.getElementById('response').style.visibility = "visible";

    if (tab.url.includes("mail.google.com")) {
        console.log("test");

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"]
        }, async () => {
            const backendURL = await getNgrokPublicUrl();
            
            if (backendURL) {
                const result = await extractEmailContent(backendURL);
                let message = document.getElementById('message');
                let loading = document.getElementById('loading');

                if (loading) loading.style.visibility = "hidden";

                if (result.error) {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = `Error: ${result.error}`;
                    }
                } else if (result.classified === "Spam") {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = "Detected Spam, act with caution";
                    }
                } else if (result.classified === "Not Spam") {
                    if (message) {
                        message.style.color = "green";
                        message.innerText = "Detected Not Spam! But be careful";
                    }
                } else {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = "ERROR: Unexpected response";
                    }
                }
            } else {
                let message = document.getElementById('message');
                if (message) {
                    message.style.color = "red";
                    message.innerText = "Error: Could not fetch ngrok URL";
                }
            }
        });
    } else {
        alert("Please open Gmail to retrieve email content.");
    }
});

async function getNgrokPublicUrl() {
    try {
        const res = await fetch('http://127.0.0.1:4040/api/tunnels');
        const data = await res.json();
        const httpsTunnel = data.tunnels.find(t => t.proto === 'https');
        return httpsTunnel ? httpsTunnel.public_url : null;
    } catch (err) {
        console.error('Error fetching ngrok URL:', err);
        return null;
    }
}

async function extractEmailContent(backendURL) {
    try {
        const emailBody = document.querySelector(".a3s");
        if (!emailBody) {
            return { error: "Could not find email content" };
        }

        const response = await fetch(`${backendURL}/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain',
            },
            body: emailBody.textContent,
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error making request:', error);
        return { error: error.message };
    }
}
