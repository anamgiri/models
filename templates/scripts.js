document.addEventListener("DOMContentLoaded", function() {
    const navItems = [
        { name: 'Home', href: 'index.html' },
        { name: 'About', href: 'about.html' },
        {
            name: 'Weekly Report', dropdown: true, items: [
                { name: 'Week 1', href: 'week1.html' },
                { name: 'Week 2', href: 'week2.html' },
                { name: 'Week 3', href: 'week3.html' },
                { name: 'Week 4', href: 'week4.html' },
                { name: 'Week 5', href: 'week5.html' },
                { name: 'Week 6', href: 'week6.html' }
            ]
        },
        {
            name: 'Models', dropdown: true, items: [
                { name: 'Model 1', href: 'model1.html' },
                { name: 'Model 2', href: 'model2.html' },
                { name: 'Model 3', href: 'model3.html' }
            ]
        },
        { name: 'Team', href: 'team.html' }
    ];

    const navbarContent = document.getElementById('navbarContent');
    const currentUrl = window.location.pathname.split('/').pop();

    navItems.forEach(item => {
        if (item.dropdown) {
            const dropdown = document.createElement('li');
            dropdown.className = 'nav-item dropdown';
            dropdown.innerHTML = `
                <a class="nav-link dropdown-toggle" href="#" id="${item.name.replace(' ', '')}Dropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                    ${item.name}
                </a>
                <div class="dropdown-menu" aria-labelledby="${item.name.replace(' ', '')}Dropdown">
                    ${item.items.map(subItem => `<a class="dropdown-item" href="${subItem.href}">${subItem.name}</a>`).join('')}
                </div>
            `;
            navbarContent.appendChild(dropdown);

            let hideTimeout;

            dropdown.addEventListener('mouseenter', function() {
                clearTimeout(hideTimeout);
                this.classList.add('show');
                this.querySelector('.dropdown-menu').classList.add('show');
            });

            dropdown.addEventListener('mouseleave', function() {
                hideTimeout = setTimeout(() => {
                    this.classList.remove('show');
                    this.querySelector('.dropdown-menu').classList.remove('show');
                }, 5000); // 5 seconds delay
            });

            dropdown.querySelector('.nav-link.dropdown-toggle').addEventListener('click', function(event) {
                event.preventDefault();
                clearTimeout(hideTimeout);
                dropdown.classList.add('show');
                dropdown.querySelector('.dropdown-menu').classList.add('show');
            });

            // Ensure dropdown stays open when clicking on dropdown items
            dropdown.querySelectorAll('.dropdown-item').forEach(function(subItem) {
                subItem.addEventListener('click', function(event) {
                    // Navigate to the href of the clicked item
                    window.location.href = subItem.href;
                });
            });

            item.items.forEach(subItem => {
                if (currentUrl === subItem.href) {
                    dropdown.classList.add('active');
                }
            });
        } else {
            const navItem = document.createElement('li');
            navItem.className = 'nav-item';
            navItem.innerHTML = `<a class="nav-link" href="${item.href}">${item.name}</a>`;
            navbarContent.appendChild(navItem);

            if (currentUrl === item.href) {
                navItem.classList.add('active');
            }
        }
    });
});
