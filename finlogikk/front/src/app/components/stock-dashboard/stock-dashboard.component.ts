import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-stock-dashboard',
  templateUrl: './stock-dashboard.component.html',
  styleUrls: ['./stock-dashboard.component.css']
})
export class StockDashboardComponent {

  streamlitUrl: SafeResourceUrl;

  constructor(private sanitizer: DomSanitizer) {
    this.streamlitUrl = this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8502');
  }

}
