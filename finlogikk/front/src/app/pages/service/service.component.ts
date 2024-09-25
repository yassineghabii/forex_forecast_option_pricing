import { Component } from '@angular/core';

@Component({
  selector: 'app-service',
  templateUrl: './service.component.html',
  styleUrls: ['./service.component.css'],
})
export class ServiceComponent {
  servcies: any = [
    {
      icon: 'fa fa-user-tie fs-4',
      heading: 'Business Planning',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-chart-pie fs-4',
      heading: 'Stretagic Planning',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-chart-line fs-4',
      heading: 'Market Analysis',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-chart-area fs-4',
      heading: 'Financial Analaysis',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-balance-scale fs-4',
      heading: 'legal Advisory',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-house-damage fs-4',
      heading: 'Tax & Insurance',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
  ];
}
