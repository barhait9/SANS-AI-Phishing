document.getElementById("getEmailButton").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    document.getElementById('response').style.visibility = "visible";
    if (tab.url.includes("mail.google.com")) {
        console.log("test");
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: extractEmailContent
        }, (results) =>{
            if (results && results[0] && results[0].result) {
                const result = results[0].result;
                let message = document.getElementById('message');
                let loading = document.getElementById('loading');
                
                if (loading) loading.style.visibility = "hidden";
                
                if (result.error) {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = `Error:${result.error}`;
                    }
                } else if (result.classified === "Spam") {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = "Detected Spam act with caution";
                    }
                } else if (result.classified === "Not Spam") {
                    if (message) {
                        message.style.color = "green";
                        message.innerText = "Detected Not Spam! but be careful";
                    }
                } else {
                    if (message) {
                        message.style.color = "red";
                        message.innerText = "ERROR";
                    }
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

async function extractEmailContent() {
    try {
        const backendURL = `https://email-api-img-673917483321.europe-west2.run.app`;
        if(!backendURL){
            return {error: "ngrok hasn't returned a url"}
        }
        const emailBody = await document.querySelector(".a3s");
        if (!emailBody) {
            return { error: "Could not find email content" };
        }
        return new Promise((resolve, _) => {
            fetch(`${backendURL}/verify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/plain',
                },
                body: emailBody.textContent,
            })
            .then(response => response.json())
            .then(result => {
                resolve(result);
            })
            .catch(error => {
                console.error('Error making request:', error);
                resolve({ error: error.message });  
            });
        });
    } catch (error) {
        console.error('Error making request:', error);
        document.getElementById('result').textContent = `Error:${error.message}`;
    }

}

