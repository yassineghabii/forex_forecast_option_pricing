import { Component } from '@angular/core';

@Component({
  selector: 'app-chooes-us',
  templateUrl: './chooes-us.component.html',
  styleUrls: ['./chooes-us.component.css'],
})
export class ChooesUsComponent {
  servcies: any = [
    {
      icon: 'fa fa-chart-bar fs-4 text-white',
      heading: 'Financial Expertise, Simple Design',
      text: 'Our exceptional financial expertise combined with a simple, user-friendly design makes navigating and using our services accessible to both beginners and seasoned investors alike.',
    },
    {
      icon: 'fa fa-lightbulb fs-4 text-white',
      heading: 'Innovative Tools for Strategic Decisions',
      text: 'Discover cutting-edge tools designed to help you make strategic decisions. We provide advanced features to forecast market trends and maximize your investments.',
    },
    {
      icon: 'fa fa-file-alt fs-4 text-white',
      heading: 'Detailed Documentation and Service Offerings',
      text: 'Access comprehensive documentation outlining every service we offer. Whether it’s financial analysis or market predictions, we provide in-depth details to guide you in using our platform effectively.',
    },
  ];

  serviceTwo: any = [
    {
      icon: 'fa fa-bolt fs-4 text-white',
      heading: 'Real-Time Information, Instant Edge',
      text: 'Stay ahead of the market with real-time information. Our instant analysis provides you with a competitive advantage by delivering up-to-date data for well-informed decisions.',
    },
    {
      icon: 'fa fa-headset fs-4 text-white',
      heading: 'Client-Focused Support, Always Available',
      text: 'Our dedicated support team is client-centered and available 24/7 to answer your questions and address concerns. We are here to guide you through every step of your financial journey.',
    },
    {
      icon: 'fa fa-users fs-4 text-white',
      heading: 'Community Engagement, Shared Financial Wisdom',
      text: 'Join an engaged community where sharing financial knowledge is encouraged. Exchange ideas, ask questions, and learn from the experiences of others, fostering a culture of shared financial wisdom.',
    }

  ];
}
