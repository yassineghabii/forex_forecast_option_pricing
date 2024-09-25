import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-about-us-box',
  templateUrl: './about-us-box.component.html',
  styleUrls: ['./about-us-box.component.css'],
})
export class AboutUsBoxComponent implements OnInit {
  @Input() data: any;
  constructor() {}

  ngOnInit(): void {}
}
