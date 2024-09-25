import { Component } from '@angular/core';

@Component({
  selector: 'app-aboutus',
  templateUrl: './aboutus.component.html',
  styleUrls: ['./aboutus.component.css'],
})
export class AboutusComponent {
  offerUs: any[] = [
    {
      icon: 'fa fa-user-tie fs-4',
      heading: 'Business Planning',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-chart-line fs-4',
      heading: 'Financial Analaysis',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
    {
      icon: 'fa fa-balance-scale fs-4',
      heading: 'legal Advisory',
      text: 'Tempor erat elitr rebum at clita. Diam dolor ipsum amet eos erat ipsum lorem et sit sed stet lorem sit clita duo',
    },
  ];
}
