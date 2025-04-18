setTimeout(() => {
    let emailBody = document.querySelector(".a3s"); // Gmail's email body class
    if (emailBody) {
        console.log("Email detected");
    } else {
        console.log("No email content found.");
    }
}, 3000);

