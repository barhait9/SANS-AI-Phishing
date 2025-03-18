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

function extractEmailContent() {
    const emailBody = document.querySelector(".a3s");
    if (emailBody) {
        console.clear;
        console.log("Email Content:", emailBody.innerText);
        //alert("Email Retrieved: " + emailBody.innerText.substring(0, 100) + "...");
    } else {
        console.log("No email found.");
        alert("No email found.");
    }
}
