import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-service-box',
  templateUrl: './service-box.component.html',
  styleUrls: ['./service-box.component.css'],
})
export class ServiceBoxComponent implements OnInit {
  @Input() service: any;  // Déclaration de l'entrée service

  constructor() {}

  ngOnInit(): void {}
}
