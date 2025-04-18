document.getElementById("getEmailButton").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (tab.url.includes("mail.google.com")) {
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: extractEmailContent
        });
    } else {
        alert("Please open Gmail to retrieve email content.");
    }
});

async function extractEmailContent() {
    const emailBody = document.querySelector(".a3s");
    if (emailBody) {
        console.clear;
        console.log("function worked");
        try {
            const response = await fetch('http://127.0.0.1:8000/verify', {
              method: 'POST',
              headers: {
                'Content-Type': 'text/plain',
              },
              body: emailBody.innerText,  // Send the raw email content as the body
            });
        
            const result = await response.json();
            if (result.classified !== undefined) {
              document.getElementById('result').textContent = `Classified result: ${result.classified}`;
            } else {
              document.getElementById('result').textContent = `Error: ${result.error}`;
            }
          } catch (error) {
            console.error('Error making request:', error);
            document.getElementById('result').textContent = `Error: ${error.message}`;
          }
    } else {
        console.log("No email found.");
        alert("No email found.");
    }
}
